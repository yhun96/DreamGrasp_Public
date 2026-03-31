import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from .uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

@dataclass
class MultiImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: List[float] = field(default_factory=lambda: [])
    default_azimuth_deg: List[float] = field(default_factory=lambda: [])
    default_camera_distance: List[float] = field(default_factory=lambda: [])
    default_fovy_deg: float = 60.0
    image_path: List[str] = field(default_factory=lambda: [])
    instance_mask_path: List[str] = field(default_factory=lambda: [])
    depth_path: List[str] = field(default_factory=lambda: [])
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    rays_d_normalize: bool = True
    eval_mode: bool = False
    real_sim_depth_ratio: float = 3.8/1.5


class MultiImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: MultiImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        elevation_deg = torch.FloatTensor(self.cfg.default_elevation_deg)
        azimuth_deg = torch.FloatTensor(self.cfg.default_azimuth_deg)
        camera_distance = torch.FloatTensor(self.cfg.default_camera_distance)

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "B 3"] = camera_position
        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
        up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
        self.c2w: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        self.c2w4x4: Float[Tensor, "B 4 4"] = torch.cat(
            [self.c2w, torch.zeros_like(self.c2w[:, :1])], dim=1
        )
        self.c2w4x4[:, 3, 3] = 1.0

        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0,)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.focal_length = self.focal_lengths[0]
        self.set_rays()
        self.load_images()
        if len(self.cfg.instance_mask_path) > 0:
            self.load_instance_masks()
        if self.cfg.eval_mode:
            self.load_depths()
        self.prev_height = self.height

    def set_rays(self):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

        rays_o, rays_d = get_rays(
            directions,
            self.c2w,
            keepdim=True,
            noise_scale=self.cfg.rays_noise_scale,
            normalize=self.cfg.rays_d_normalize,
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx        
    
    def load_images(self):
        # load image
        self.rgb = []
        self.mask = []
        for image_path in self.cfg.image_path:
            assert os.path.exists(
                image_path
            ), f"Could not find image {image_path}!"
            rgba = cv2.cvtColor(
                cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
            )
            rgba = (
                cv2.resize(
                    rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
                ).astype(np.float32)
                / 255.0
            )
            rgb = rgba[..., :3]
            rgb: Float[Tensor, "1 H W 3"] = (
                torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
            )
            self.rgb.append(rgb)
            mask: Float[Tensor, "1 H W 1"] = (
                torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
            )
            self.mask.append(mask)
            # print(
            #     f"[INFO] multi image dataset: load image {image_path} {rgb.shape}"
            # )

        self.rgb: Float[Tensor, "B H W 3"] = torch.cat(self.rgb, dim=0)
        self.mask: Float[Tensor, "B H W 1"] = torch.cat(self.mask, dim=0)
    
    def load_instance_masks(self):
        self.instance_mask = []
        for path in self.cfg.instance_mask_path:
            instance_mask = np.load(path)
            instance_mask = cv2.resize(instance_mask.transpose(1, 2, 0).astype(float), (self.width, self.height), interpolation=cv2.INTER_AREA)
            instance_mask = instance_mask > 0.5
            instance_mask = torch.from_numpy(instance_mask).to(self.rank)
            self.instance_mask.append(instance_mask)
        self.instance_mask = torch.stack(self.instance_mask) # B H W N_instance
    
    def load_depths(self):
        self.depth = []
        for path in self.cfg.depth_path:
            depth = np.load(path)
            depth = cv2.resize(depth.astype(float), (self.width, self.height), interpolation=cv2.INTER_AREA)
            depth = torch.from_numpy(depth).to(self.rank)
            self.depth.append(depth)
        self.depth = torch.stack(self.depth) # B H W 
        self.depth *= self.cfg.real_sim_depth_ratio
    
    def get_all_images(self):
        return self.rgb

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")
        self.set_rays()
        self.load_images()
        if len(self.cfg.instance_mask_path) > 0:
            self.load_instance_masks()
        if self.cfg.eval_mode:
            self.load_depths()

class MultiImageIterableDataset(IterableDataset, MultiImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str):
        super().__init__()
        self.setup(cfg, split)
        self.len = len(self.rgb)

    def collate(self, batch):
        batch = {
            "rays_o": self.rays_o,
            "rays_d": self.rays_d,
            "mvp_mtx": self.mvp_mtx,
            "camera_positions": self.camera_position,
            "light_positions": self.light_position,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "rgb" : self.rgb,
            "mask": self.mask,
            "c2w": self.c2w4x4,
            "height": self.height,
            "width": self.width,
            "fovy": torch.cat([self.fovy] * self.len, dim=0)
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)
        if len(self.cfg.instance_mask_path) > 0:
            batch["instance_mask"] = self.instance_mask
        return batch
        
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}

class MultiImageDataset(Dataset, MultiImageDataBase):
    def __init__(self, cfg: Any, split: str):
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        return self.random_pose_generator[index]

class EvalMultiImageDataset(Dataset, MultiImageDataBase):
    def __init__(self, cfg: Any, split: str):
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        batch = {
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "camera_positions": self.camera_position[index],
            "light_positions": self.light_position[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distance[index],
            "rgb" : self.rgb[index],
            "mask": self.mask[index],
            "depth": self.depth[index],
            "c2w": self.c2w4x4[index],
            "height": self.height,
            "width": self.width,
            "fovy": self.fovy
        }
        if len(self.cfg.instance_mask_path) > 0:
            batch["instance_mask"] = self.instance_mask[index]
        return batch

@register("multi-image-datamodule")
class MultiImageDataModule(pl.LightningDataModule):
    cfg: MultiImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        super().__init__()
        self.cfg = parse_structured(MultiImageDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = MultiImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            if self.cfg.eval_mode:
                self.test_dataset = EvalMultiImageDataset(self.cfg, "test")
            else:
                self.test_dataset = MultiImageDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None):
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self):
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)
