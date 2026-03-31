import os
import random
import shutil
import pickle
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

from copy import deepcopy
import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from torch.cuda.amp import autocast
from ..utils.fusion import TSDFVolume
import open3d as o3d
from tqdm import tqdm

from ..geometry.gaussian_base import BasicPointCloud, Camera, gaussian_gradients, gaussian_3d_coeff, compute_normalized_field_gradient
from ..geometry.mesh_utils import *


@threestudio.register("dreamgrasp-system")
class DreamGrasp(BaseLift3DSystem):
	@dataclass
	class Config(BaseLift3DSystem.Config):
		freq: dict = field(default_factory=dict)
		ambient_ratio_min: float = 0.5
		back_ground_color: Tuple[float, float, float] = (1, 1, 1)
		
		zero123_guidance_type: str = ""
		zero123_guidance: dict = field(default_factory=dict)
		sds_guidance_type: str = ""
		sds_guidance: dict = field(default_factory=dict)
		
	cfg: Config

	def configure(self):
		# create geometry, material, background, renderer
		super().configure()
		# no prompt processor
		self.zero123_guidance = threestudio.find(self.cfg.zero123_guidance_type)(self.cfg.zero123_guidance)
		self.sds_guidance = threestudio.find(self.cfg.sds_guidance_type)(self.cfg.sds_guidance)
		self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
			self.cfg.prompt_processor
		)
		self.prompt_utils = self.prompt_processor()
		self.automatic_optimization = False

	def configure_optimizers(self):
		optim = self.geometry.optimizer
		if hasattr(self, "merged_optimizer"):
			return [optim]
		if hasattr(self.cfg.optimizer, "name"):
			net_optim = parse_optimizer(self.cfg.optimizer, self)
			optim = self.geometry.merge_optimizer(net_optim)
			self.merged_optimizer = True
		else:
			self.merged_optimizer = False
		return [optim]

	def on_load_checkpoint(self, checkpoint):
		num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
		pcd = BasicPointCloud(
			points=np.zeros((num_pts, 3)),
			colors=np.zeros((num_pts, 3)),
			normals=np.zeros((num_pts, 3)),
		)
		self.geometry.create_from_pcd(pcd, 10)
		self.geometry.training_setup()
		return

	def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		self.geometry.update_learning_rate(self.global_step)
		outputs = self.renderer.batch_forward(batch)
		return outputs

	def on_fit_start(self) -> None:
		super().on_fit_start()
		
		# visualize all training images
		all_images = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
		self.save_image_grid(
			"all_training_images.png",
			[
				{"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
				for image in all_images
			],
			name="on_fit_start",
			step=self.true_global_step,
		)

	def training_substep(self, batch, batch_idx, guidance: str):
		"""
		Args:
			guidance: one of "ref" (reference image supervision), "zero123"
		"""
		if guidance == "ref":
			batch["ambient_ratio"] = 1.0
			shading = "diffuse"
			batch["shading"] = shading
			if self.C(self.cfg.loss.lambda_mask_contrast) > 0 and self.true_global_step < self.cfg.freq.get("segmentation_start_iter"):
				batch["render_feature"] = True
			elif self.true_global_step >= self.cfg.freq.get("segmentation_start_iter"):
				batch["render_instance"] = True
			
		elif guidance == "zero123":
			batch = batch["random_camera"]
			batch["ambient_ratio"] = (
				self.cfg.ambient_ratio_min
				+ (1 - self.cfg.ambient_ratio_min) * random.random()
			)

		elif guidance == "sds":
			batch = batch["random_camera"]
			batch["ambient_ratio"] = (
				self.cfg.ambient_ratio_min
				+ (1 - self.cfg.ambient_ratio_min) * random.random()
			)
			batch["render_color"] = False
			batch["render_instance"] = True

		out = self(batch)
			
		loss_prefix = f"loss_{guidance}_"
		loss_terms = {}

		def set_loss(name, value):
			loss_terms[f"{loss_prefix}{name}"] = value

		guidance_eval = (
			guidance == "zero123"
			and self.cfg.freq.guidance_eval > 0
			and self.true_global_step % self.cfg.freq.guidance_eval == 0
		)

		if guidance == "ref":
			gt_mask = batch["mask"]
			gt_rgb = batch["rgb"]

			# color loss
			# if self.true_global_step < self.cfg.freq.get("segmentation_start_iter"):
			gt_rgb = gt_rgb * gt_mask.float()
			set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"] * gt_mask.float()))

			# mask loss
			set_loss("mask", F.mse_loss(gt_mask.float(), out["comp_mask"]))
			
			# instance-wise RGB loss
			if self.true_global_step >= self.cfg.freq.get("segmentation_start_iter"):
				gt_instance_mask = batch["instance_mask"]
				set_loss("instancewise_rgb", F.mse_loss(
					gt_rgb.unsqueeze(1).repeat(1, gt_instance_mask.shape[-1], 1, 1, 1)[gt_instance_mask.permute(0, 3, 1, 2)],
					out["comp_instancewise_rgb"][gt_instance_mask.permute(0, 3, 1, 2)])
				)
				
				# if self.true_global_step % 10 == 0:
				#     self.save_image_grid(
				#         f"it{self.true_global_step}_instance_rgb.png",
				#         (
				#         [
				#             {
				#                 "type": "rgb",
				#                 "img": instance_rgb,
				#                 "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
				#             }
				#             for instance_rgb in out["comp_instancewise_rgb"][0]
				#         ]
				#         if "comp_instancewise_rgb" in out
				#         else []
				#     )
				#     )
			
			# constrastive loss
			if (self.C(self.cfg.loss.lambda_mask_contrast) > 0) and (self.true_global_step < self.cfg.freq.get("segmentation_start_iter")):
				gt_instance_mask = batch["instance_mask"]
				rendered_mask = out["comp_another_feature"]
				set_loss("mask_contrast", mask_contrastive_loss(rendered_mask, gt_instance_mask))
			
		elif guidance == "zero123":
			# zero123
			guidance_out = self.zero123_guidance(
				out["comp_rgb"],
				**batch,
				rgb_as_latents=False,
				guidance_eval=guidance_eval,
			)
			# claforte: TODO: rename the loss_terms keys
			set_loss("zero123", guidance_out["loss_sds"])
		
		elif guidance == "sds":
			guidance_inp = out["comp_instancewise_rgb"]
			loss_sds = 0
			for i in range(guidance_inp.shape[1]):
				if isinstance(self.prompt_utils, List):
					guidance_out = self.sds_guidance(
						guidance_inp[:,i,...], self.prompt_utils[i], **batch, rgb_as_latents=False
					)
				else:
					guidance_out = self.sds_guidance(
						guidance_inp[:,i,...], self.prompt_utils, **batch, rgb_as_latents=False
					)
				loss_sds += guidance_out["loss_sds"]
			set_loss("sds", loss_sds)

		if (self.C(self.cfg.loss.lambda_normal_smooth) > 0) and (guidance != "sds"):
			if "comp_normal" not in out:
				raise ValueError(
					"comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
				)
			normal = out["comp_normal"]
			set_loss(
				"normal_smooth",
				(normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
				+ (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
			)
		
		if self.C(self.cfg.loss["lambda_opacity"]) > 0.0:
			scaling = self.geometry.get_scaling.norm(dim=-1)
			loss_opacity = (
				scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
			).sum()
			set_loss("opacity", loss_opacity)
		
		if (self.C(self.cfg.loss.lambda_spatial_similarity) > 0) and (self.true_global_step < self.cfg.freq.get("segmentation_start_iter")):
			spatial_reg = spatial_loss(self.geometry.get_xyz, self.geometry.get_another_feature)
			set_loss("spatial_similarity", spatial_reg)
			
		if (self.C(self.cfg.loss.lambda_feature_gradient) > 0) and (self.true_global_step < self.cfg.freq.get("segmentation_start_iter")):
			feature_gradient = feature_gradient_loss(
				self.geometry.get_xyz.detach(),
				self.geometry.get_covariance().detach(),
				self.geometry.get_opacity.detach(),
				self.geometry.get_another_feature)
			set_loss("feature_gradient", feature_gradient)
			
		loss = 0.0
		for name, value in loss_terms.items():
			self.log(f"train/{name}", value)
			if name.startswith(loss_prefix):
				loss_weighted = value * self.C(
					self.cfg.loss[name.replace(loss_prefix, "lambda_")]
				)
				self.log(f"train/{name}_w", loss_weighted)
				self.loss_dict[name.replace(loss_prefix,"")] = loss_weighted
				loss += loss_weighted

		for name, value in self.cfg.loss.items():
			self.log(f"train_params/{name}", self.C(value))

		self.log(f"train/loss_{guidance}", loss)

		out.update({"loss": loss})
		return out

	@torch.no_grad()
	def get_mask_features(self, mask_feature, gt_mask):
		feature_list = []
		for i in range(gt_mask.shape[-1]):
			if gt_mask[...,i].sum() == 0:
				feature_list.append(torch.zeros_like(mask_feature[0, 0]))
			else:
				mask_feature_new = deepcopy(mask_feature)
				mask_feature_new[~gt_mask[...,i]] = 0
				selected_feature = (mask_feature[gt_mask[...,i]]).mean(dim=0)
				feature_list.append(F.normalize(selected_feature, dim=-1))
		feature_list = torch.stack(feature_list)
		return feature_list
	
	@torch.no_grad()
	def segment_instances(self, batch):
		batch["ambient_ratio"] = 1.0
		shading = "diffuse"
		batch["shading"] = shading
		batch["render_feature"] = True
		out = self(batch)

		bs = len(out["comp_another_feature"])
		for representative_idx in range(bs):
			mask_feature = out["comp_another_feature"][representative_idx] * out["comp_mask"][representative_idx]
			gt_mask = batch['instance_mask'][representative_idx]
			if torch.all(gt_mask.sum(dim=[0,1]) != 0):
				break
		
		feature_list = self.get_mask_features(mask_feature, gt_mask)
		self.geometry.classify_by_feature(feature_list)
		
		# align instance channel by feature similarity
		for i in range(bs):
			if i != representative_idx:
				mask_feature = out["comp_another_feature"][i] * out["comp_mask"][i]
				gt_mask = batch['instance_mask'][i]
				feature_list_ = self.get_mask_features(mask_feature, gt_mask)
				feature_dist_matrix = torch.cdist(feature_list, feature_list_)
				_, col_ind = linear_sum_assignment(feature_dist_matrix.detach().cpu().numpy())
				self.trainer.train_dataloader.dataset.instance_mask[i,...] = self.trainer.train_dataloader.dataset.instance_mask[i,...,col_ind]
	
	def training_step(self, batch, batch_idx):
		opt = self.optimizers()

		if self.cfg.freq.get("segmentation_start_iter") == self.true_global_step:
			self.segment_instances(batch)
		
		do_ref = True  
		do_zero123 = self.true_global_step <= self.cfg.freq.get("zero123_end_iter") and self.true_global_step >= self.cfg.freq.get("zero123_start_iter", 0)
		do_sds = self.true_global_step > self.cfg.freq.get("sds_start_iter")
		
		if self.cfg.freq.get("alter_after_seg", False) and self.true_global_step >= self.cfg.freq.get("segmentation_start_iter"):
			if self.true_global_step % 2 == 0:
				do_ref = True  
				do_zero123 = self.true_global_step <= self.cfg.freq.get("zero123_end_iter")
				do_sds = False
			else:
				do_ref = False  
				do_zero123 = False
				do_sds = self.true_global_step > self.cfg.freq.get("sds_start_iter")
		
		total_loss = 0.0
		self.loss_dict = {}
		
		instancewise_visibility_filter = []
		instancewise_viewspace_points = []
		visibility_filter  = []
		radii = []
		viewspace_point_tensor = []
		
		if do_ref:
			out = self.training_substep(batch, batch_idx, guidance="ref")
			total_loss += out["loss"]
			visibility_filter += out["visibility_filter"]
			radii += out["radii"]
			viewspace_point_tensor = out["viewspace_points"]
			if "instancewise_visibility_filter" in out:
				instancewise_visibility_filter += out["instancewise_visibility_filter"]
				instancewise_viewspace_points += out["instancewise_viewspace_points"]
		
		if do_zero123:
			out = self.training_substep(batch, batch_idx, guidance="zero123")
			total_loss += out["loss"]
			visibility_filter += out["visibility_filter"]
			radii += out["radii"]
			viewspace_point_tensor += out["viewspace_points"]
		
		if do_sds:
			out = self.training_substep(batch, batch_idx, guidance="sds")
			total_loss += out["loss"]
			instancewise_visibility_filter += out["instancewise_visibility_filter"]
			instancewise_viewspace_points += out["instancewise_viewspace_points"]

		self.log("train/loss", total_loss)
		self.log_dict(
			self.loss_dict,
			prog_bar=True)

		total_loss.backward()
		iteration = self.global_step
		self.geometry.update_states(
			iteration,
			visibility_filter,
			radii,
			viewspace_point_tensor,
			instancewise_visibility_filter,
			instancewise_viewspace_points,
			batch['width'],
			batch['height']
		)

		opt.step()
		opt.zero_grad(set_to_none=True)

		if (self.true_global_step > self.cfg.freq.get("segmentation_start_iter")) and \
			(self.true_global_step % self.cfg.freq.get("instance_viewpose_update_iter") == 0):
			bbox_min, bbox_max = self.geometry.get_bboxs_of_instances()
			self.trainer.train_dataloader.dataset.random_pose_generator.update_instancewise_camera(bbox_min, bbox_max)
			self.trainer.val_dataloaders.dataset.random_pose_generator.update_instancewise_camera(bbox_min, bbox_max)
  
		return {"loss": total_loss}

	def validation_step(self, batch, batch_idx):
		batch["render_feature"] = True
		if self.true_global_step > self.cfg.freq.get("segmentation_start_iter"):
			batch["render_instance"] = True
		out = self(batch)
		self.save_image_grid(
			f"it{self.true_global_step}-val/{batch['index'][0]}.png",
			(
				[
					{
						"type": "rgb",
						"img": batch["rgb"][0],
						"kwargs": {"data_format": "HWC"},
					}
				]
				if "rgb" in batch
				else []
			)
			+ [
				{
					"type": "rgb",
					"img": out["comp_rgb"][0],
					"kwargs": {"data_format": "HWC"},
				},
			]
			+ (
				[
					{
						"type": "rgb",
						"img": out["comp_normal"][0],
						"kwargs": {"data_format": "HWC", "data_range": (0, 1)},
					}
				]
				if "comp_normal" in out
				else []
			)
			+ (
				[
					{
						"type": "grayscale",
						"img": out["comp_depth"][0],
						"kwargs": {},
					}
				]
				if "comp_depth" in out
				else []
			)
			# + (
			#     [
			#         {
			#             "type": "grayscale",
			#             "img": out["comp_mask"][0],
			#             "kwargs": {},
			#         }
			#     ]
			#     if "comp_mask" in out
			#     else []
			# )
			+ (
				[
					{
						"type": "rgb",
						"img": out["comp_another_feature"][0][...,:3] * out["comp_mask"][0] * 0.5 + 0.5,
						"kwargs": {"data_format": "HWC", "data_range": (0, 1)},
					}
				]
				if "comp_another_feature" in out
				else []
			)
			+ (
				[
					{
						"type": "rgb",
						"img": instance_rgb,
						"kwargs": {"data_format": "HWC", "data_range": (0, 1)},
					}
					for instance_rgb in out["comp_instancewise_rgb"][0]
				]
				if "comp_instancewise_rgb" in out
				else []
			),
			# claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
			name=f"validation_step_batchidx_{batch_idx}"
			if batch_idx in [0, 7, 15, 23, 29]
			else None,
			step=self.true_global_step,
		)

	def on_validation_epoch_end(self):
		filestem = f"it{self.true_global_step}-val"
		self.save_img_sequence(
			filestem,
			filestem,
			"(\d+)\.png",
			save_format="mp4",
			fps=30,
			name="validation_epoch_end",
			step=self.true_global_step,
		)
		shutil.rmtree(
			os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
		)

	def test_step(self, batch, batch_idx):
		batch["render_feature"] = True
		if self.true_global_step > self.cfg.freq.get("segmentation_start_iter"):
			batch["render_instance"] = True
			batch["render_instance_depth"] = True
		out = self(batch)

		# self.save_data(f"test_raw/{batch['index'][0]}.npz", out)
		self.save_image_grid(
			f"it{self.true_global_step}-test/{batch['index'][0]}.png",
			(
				[
					{
						"type": "rgb",
						"img": batch["rgb"][0],
						"kwargs": {"data_format": "HWC"},
					}
				]
				if "rgb" in batch
				else []
			)
			+ [
				{
					"type": "rgb",
					"img": out["comp_rgb"][0],
					"kwargs": {"data_format": "HWC"},
				},
			]
			+ (
				[
					{
						"type": "rgb",
						"img": out["comp_normal"][0],
						"kwargs": {"data_format": "HWC", "data_range": (0, 1)},
					}
				]
				if "comp_normal" in out
				else []
			)
			+ (
				[
					{
						"type": "grayscale",
						"img": out["comp_depth"][0],
						"kwargs": {},
					}
				]
				if "comp_depth" in out
				else []
			)
			# + (
			#     [
			#         {
			#             "type": "grayscale",
			#             "img": out["comp_mask"][0],
			#             "kwargs": {},
			#         }
			#     ]
			#     if "comp_mask" in out
			#     else []
			# )
			+ (
				[
					{
						"type": "rgb",
						"img": out["comp_another_feature"][0][...,:3] * out["comp_mask"][0] * 0.5 + 0.5,
						"kwargs": {"data_format": "HWC", "data_range": (0, 1)},
					}
				]
				if "comp_another_feature" in out
				else []
			)
			+ (
				[
					{
						"type": "rgb",
						"img": instance_rgb,
						"kwargs": {"data_format": "HWC", "data_range": (0, 1)},
					}
					for instance_rgb in out["comp_instancewise_rgb"][0]
				]
				if "comp_instancewise_rgb" in out
				else []
			),
			name="test_step",
			step=self.true_global_step,
		)
		
		tsdf_data_dict = {
			"depth" : out["comp_depth"][0].squeeze().detach().cpu().numpy(),
			"c2w" : batch["c2w"].squeeze().detach().cpu().numpy()
		}
		if "comp_instancewise_depth" in out:
			tsdf_data_dict["instancewise_depth"] = out["comp_instancewise_depth"][0].squeeze().detach().cpu().numpy()
		
		self.save_data(f"depth_for_tsdf/{batch['index'][0]}.npz", tsdf_data_dict)

	def on_test_epoch_end(self):
		self.save_img_sequence(
			f"it{self.true_global_step}-test",
			f"it{self.true_global_step}-test",
			"(\d+)\.png",
			save_format="mp4",
			fps=30,
			name="test",
			step=self.true_global_step,
		)
	
		save_path = self.get_save_path("point_cloud.ply")
		self.geometry.save_ply(save_path)
		
		shutil.rmtree(
			os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test")
		)
		try:
			self.make_tsdf_mesh()
		except:
			pass
		
		if os.path.exists(os.path.join(self.get_save_dir(), "depth_for_tsdf")):
			shutil.rmtree(
				os.path.join(self.get_save_dir(), "depth_for_tsdf")
			)   
		
	def make_tsdf_mesh(self):
		if os.path.exists(os.path.join(self.get_save_dir(), f"it{self.true_global_step}-tsdf_mesh")):
			print("folder already exists. skip tsdf fusion.")
			return
		else:
			os.makedirs(os.path.join(self.get_save_dir(), f"it{self.true_global_step}-tsdf_mesh"))
		
		depth_dir = os.path.join(self.get_save_dir(), "depth_for_tsdf")
		depth_paths = sorted(
			os.listdir(depth_dir),
			key=lambda x: int(x.split('.')[0]))
		depth_paths = [os.path.join(depth_dir, d) for d in depth_paths]
		
		depths = []
		instancewise_depths = []
		camera_poses = []
		for path in depth_paths:
			data = np.load(path)
			depths.append(data["depth"].squeeze())
			camera_poses.append(data["c2w"] @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]])) # for open3d camera frame
			if "instancewise_depth" in data:
				instancewise_depths.append(data["instancewise_depth"].squeeze())
		
		depths = np.stack(depths)
		camera_poses = np.stack(camera_poses)
		if len(instancewise_depths) > 0:
			instancewise_depths = np.stack(instancewise_depths, axis=1)
		
		fx = depths.shape[-2] / (2 * np.tan(20/2 /180*np.pi))
		fy = depths.shape[-1] / (2 * np.tan(20/2 /180*np.pi))
		cx = depths.shape[-2]/2
		cy = depths.shape[-1]/2
		intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
		xyzs = self.geometry.get_xyz.detach().cpu().numpy()
		resolution=128
		voxel_size = (xyzs.max(axis=0) - xyzs.min(axis=0)).min()/resolution
		
		verts, faces = self.tsdf_fusion(depths, camera_poses, intrinsic, np.stack([xyzs.min(axis=0), xyzs.max(axis=0)]).T, voxel_size)
		scene_mesh = self.postprocess_mesh(verts, faces)
		
		o3d.io.write_triangle_mesh(os.path.join(self.get_save_dir(), f"it{self.true_global_step}-tsdf_mesh","model.obj"), scene_mesh)

		for idx, instancewise_depth in enumerate(instancewise_depths):
			try:
				verts, faces = self.tsdf_fusion(instancewise_depth, camera_poses, intrinsic, np.stack([xyzs.min(axis=0), xyzs.max(axis=0)]).T, voxel_size)
				instance_mesh = self.postprocess_mesh(verts, faces)
				o3d.io.write_triangle_mesh(os.path.join(self.get_save_dir(), f"it{self.true_global_step}-tsdf_mesh",f"model_{idx}.obj"), instance_mesh)
			except:
				pass

	def tsdf_fusion(self, depths, camera_poses, intrinsic, vol_bnds, voxel_size):
		# tsdf initialize
		tsdf = TSDFVolume(vol_bnds, voxel_size, use_gpu=True)
		
		for depth, camera_pose in tqdm(zip(depths, camera_poses), desc="tsdf fusion..."):
			color_img = np.ones((depth.shape[0], depth.shape[1], 3)) * 0.7
			depth[depth < 0.1] = 20.
			tsdf.integrate(color_img, depth, intrinsic, camera_pose, obs_weight=1.)
		
		verts, faces, _, _ = tsdf.get_mesh()

		return verts, faces

	def postprocess_mesh(self, verts, faces, decimate_target=1e5):
		# try:
		#     verts, faces = clean_mesh(
		#     verts, faces, remesh=True, remesh_size=0.015
		#     )
		#     if decimate_target > 0 and faces.shape[0] > decimate_target:
		#         verts, faces = decimate_mesh(verts, faces, decimate_target)
		# except:
		#     pass
		mesh = o3d.geometry.TriangleMesh()
		mesh.vertices = o3d.utility.Vector3dVector(verts)
		mesh.triangles = o3d.utility.Vector3iVector(faces)
		return mesh


	
# from https://github.com/MyrnaCCS/contrastive-gaussian-clustering/blob/main/utils/loss_utils.py#L104
def spatial_loss(xyz, features, k_pull=2, k_push=5, lambda_pull=0.05, lambda_push=0.15, max_points=200000, sample_size=800):
	"""
	Compute the spatial-similarity regularization loss for a 3D point cloud using Top-k neighbors and Top-k distant elements
	
	:param xyz: Tensor of shape (N, D), where N is the number of points and D is the dimensionality.
	:param features: Tensor of shape (N, C), where C is the dimensionality of these features.
	:param k_pull: Number of neighbors to consider.
	:param k_push: Number of remote elements to consider.
	:param lambda_pull: Weighting factor for the loss.
	:param lambda_push: Weighting factor for the loss.
	:param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
	:param sample_size: Number of points to randomly sample for computing the loss.
	
	:return: Computed loss value.
	"""
	# Conditionally downsample if points exceed max_points
	if xyz.size(0) > max_points:
		indices = torch.randperm(xyz.size(0))[:max_points]
		xyz = xyz[indices]
		features = features[indices]

	# Randomly sample points for which we'll compute the loss
	indices = torch.randperm(xyz.size(0))[:sample_size]
	sample_xyz = xyz[indices]
	sample_preds = features[indices]

	# Compute top-k nearest neighbors directly in PyTorch
	dists = torch.cdist(sample_xyz, xyz)  # Compute pairwise distances
	_, neighbor_indices_tensor = dists.topk(k_pull, largest=False)  # Get top-k smallest distances

	# Compute top-k farest gaussians
	_, faraway_indices_tensor = dists.topk(k_push, largest=True)  # Get top-k bigest distances 

	# Fetch neighbor features using indexing
	neighbor_preds = features[neighbor_indices_tensor]

	# Fetch remote features using indexing
	faraway_preds = features[faraway_indices_tensor]

	# Compute cosine similarity
	cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
	pull_loss = cos(sample_preds.unsqueeze(1).expand(-1, k_pull, -1), neighbor_preds) #more similar of they are close
	push_loss = cos(sample_preds.unsqueeze(1).expand(-1, k_push, -1), faraway_preds) #less similar if they are far away
	
	# Total loss
	# loss = lambda_pull * torch.sigmoid(1.0 - torch.mean(pull_loss[..., 0].reshape(-1), dim=-1)) + \
	#     lambda_push * torch.sigmoid(torch.mean(push_loss[..., 0].reshape(-1), dim=-1))
	loss = lambda_pull * (1.0 - pull_loss).mean() + lambda_push * (1 + push_loss).mean()
	return loss

def feature_gradient_loss(means, covs, opac, features, xyz_sample_size=800, field_sample_size=8000):
	N = means.size(0)
	xyz_indices = torch.randperm(N)[:xyz_sample_size]
	field_indices = torch.randperm(N)[:field_sample_size]
	xyzs = means[xyz_indices]
	means = means[field_indices]
	covs = covs[field_indices]
	opac = opac[field_indices]
	features = features[field_indices]
	w, w_grads = gaussian_gradients(xyzs, means, covs)    
	f = w @ (features * opac)
	f_grad = (w_grads.unsqueeze(-2) * (features * opac).reshape(1, -1, 3, 1)).sum(1)
	grad_F_normalized = compute_normalized_field_gradient(f, f_grad).squeeze()
	
	# return grad_F_normalized.norm(dim=(1,2)).mean()
	Lambda_inv = torch.diag(torch.tensor([0.01, 1., 1.])).to(means.device)
	density_grad = (w_grads * opac.reshape(1, -1, 1)).sum(1)
	normals = torch.nn.functional.normalize(density_grad, dim=-1)
	R = generate_rotation_matrices_from_normals(normals)
	G_inv = R @ Lambda_inv @ R.permute(0, 2, 1)
	
	# for numerical stability, drop degenerated rotation matrices
	mask = (w_grads * opac.reshape(1, -1, 1)).sum(1).norm(dim=-1) > 0.01
	G_inv = G_inv[mask]
	grad_F_normalized = grad_F_normalized[mask]
	
	return torch.diagonal(G_inv.detach() @ grad_F_normalized.permute(0, 2, 1) @ grad_F_normalized, dim1=-2, dim2=-1).sum(dim=-1).mean()

def mask_contrastive_loss(rendered_mask, gt_instance_mask):
	B, H, W, _ = gt_instance_mask.shape
	D = rendered_mask.shape[-1]
	
	loss = 0
	for i in range(B):
		Np = gt_instance_mask[[i]].sum(dim=(1,2))
		mask_existence_bool = (Np != 0).squeeze()
		N = mask_existence_bool.sum().item()
		Np = Np[:, mask_existence_bool]
		gt_instance_mask_i = gt_instance_mask[[i]][..., mask_existence_bool]
		masked_feature = rendered_mask[[i]].reshape(1, H, W, 1, D).repeat(1, 1, 1, N, 1) * gt_instance_mask_i.reshape(1, H, W, N, 1).float()
		feature_mean = masked_feature.sum(dim=(1, 2)) / Np.reshape(1, N, 1)
		feature_var = masked_feature - feature_mean.reshape(1, 1, 1, N, D)
		feature_var[~gt_instance_mask_i] = 0
		temperature = (feature_var.norm(dim=-1).sum(dim=(1,2))/(Np*torch.log(Np+100))).detach()
		temperature = torch.clip(temperature * 10, min=0.1, max=1.0)
		pos = ((masked_feature * feature_mean.reshape(1, 1, 1, N, D)).sum(dim=-1) / temperature.reshape(1, 1, 1, N)).sum() / (W*H*N)
		neg = torch.log(
			torch.exp(
				(masked_feature.reshape(1, H, W, N, 1, D) * feature_mean.reshape(1, 1, 1, 1, N, D)).sum(dim=-1) / temperature.reshape(1, 1, 1, 1, N)
			).sum(dim=-1)[gt_instance_mask_i]
		).sum() / (W*H*N)
		loss += neg - pos
	
	return loss

def generate_rotation_matrices_from_normals(normals):
    """
    Generate Bx3x3 rotation matrices where the first column is the given normal vectors.
    
    :param normals: Bx3 tensor of normal vectors
    :return: Bx3x3 tensor of rotation matrices
    """
    B = normals.size(0)
    rotation_matrices = torch.zeros((B, 3, 3), dtype=normals.dtype, device=normals.device)

    # Assign the normalized normal vectors to the first column of each 3x3 matrix.
    rotation_matrices[:, :, 0] = normals

    # Generate a second vector that is not collinear with the normal vector.
    arbitrary_vec = torch.tensor([1, 0, 0], dtype=normals.dtype, device=normals.device)
    second_vec = torch.cross(normals, arbitrary_vec.expand_as(normals))
    
    # If any vector is too close to [1, 0, 0], switch to [0, 1, 0].
    mask = torch.norm(second_vec, dim=1) < 1e-6
    second_vec[mask] = torch.cross(normals[mask], torch.tensor([0, 1, 0], dtype=normals.dtype, device=normals.device).expand_as(normals[mask]))
    
    # Normalize the second vector to ensure it's a unit vector.
    second_vec = second_vec / torch.norm(second_vec, dim=1, keepdim=True)

    # Assign the second vector to the second column.
    rotation_matrices[:, :, 1] = second_vec

    # Compute the third vector as the cross product of the first two vectors (ensures orthogonality).
    third_vec = torch.cross(rotation_matrices[:, :, 0], rotation_matrices[:, :, 1])
    
    # Assign the third vector to the third column.
    rotation_matrices[:, :, 2] = third_vec

    return rotation_matrices