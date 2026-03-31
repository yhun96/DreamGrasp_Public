import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from gsplat import rasterization

from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

from .gaussian_batch_renderer import GaussianBatchRenderer

class Depth2Normal(torch.nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.delzdelxkernel = torch.tensor(
			[
				[0.00000, 0.00000, 0.00000],
				[-1.00000, 0.00000, 1.00000],
				[0.00000, 0.00000, 0.00000],
			]
		)
		self.delzdelykernel = torch.tensor(
			[
				[0.00000, -1.00000, 0.00000],
				[0.00000, 0.00000, 0.00000],
				[0.0000, 1.00000, 0.00000],
			]
		)

	def forward(self, x):
		B, C, H, W = x.shape
		delzdelxkernel = self.delzdelxkernel.view(1, 1, 3, 3).to(x.device)
		delzdelx = F.conv2d(
			x.reshape(B * C, 1, H, W), delzdelxkernel, padding=1
		).reshape(B, C, H, W)
		delzdelykernel = self.delzdelykernel.view(1, 1, 3, 3).to(x.device)
		delzdely = F.conv2d(
			x.reshape(B * C, 1, H, W), delzdelykernel, padding=1
		).reshape(B, C, H, W)
		normal = -torch.cross(delzdelx, delzdely, dim=1)
		return normal

@threestudio.register("gsplat-unified")
class DiffGaussian(Rasterizer, GaussianBatchRenderer):
	@dataclass
	class Config(Rasterizer.Config):
		debug: bool = False
		invert_bg_prob: float = 1.0
		back_ground_color: Tuple[float, float, float] = (1, 1, 1)

	cfg: Config

	def configure(
		self,
		geometry: BaseGeometry,
		material: BaseMaterial,
		background: BaseBackground,
	) -> None:
		threestudio.info(
			"[Note] Gaussian Splatting doesn't support material and background now."
		)
		super().configure(geometry, material, background)
		self.normal_module = Depth2Normal()
		self.background_tensor = torch.tensor(
			self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
		)

	def forward(
		self,
		viewpoint_camera,
		bg_color: torch.Tensor,
		scaling_modifier=1.0,
		override_color=None,
		render_color=True,
		render_feature=False,
		render_instance=False,
		render_instance_depth=False,
		instance_viewpoint_cams=None,
		**kwargs
	) -> Dict[str, Any]:
		"""
		Render the scene.

		Background tensor (bg_color) must be on GPU!
		"""

		if self.training:
			invert_bg_color = np.random.rand() > self.cfg.invert_bg_prob
		else:
			invert_bg_color = True
		bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)
		
		pc = self.geometry

		# Set up camera parameters
		tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
		tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
		focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
		focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
		K = torch.tensor(
			[
				[focal_length_x, 0, viewpoint_camera.image_width / 2.0],
				[0, focal_length_y, viewpoint_camera.image_height / 2.0],
				[0, 0, 1],
			],
			device="cuda",
		)

		means3D = pc.get_xyz
		opacity = pc.get_opacity

		# If precomputed 3d covariance is provided, use it. If not, then it will be computed from
		# scaling / rotation by the rasterizer.
		scales = None
		rotations = None
		scales = pc.get_scaling
		rotations = pc.get_rotation

		# If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
		# from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
		
		if override_color is None:
			shs = pc.get_features
			sh_degree=pc.active_sh_degree
		else:
			shs = override_color
			sh_degree = None
		viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device) 
		
		out = {}

		if render_color:        
			# Rasterize visible Gaussians to image, obtain their radii (on screen).

			render_RGBD, rendered_alpha, info = rasterization(
			means=means3D,  # [N, 3]
			quats=rotations,  # [N, 4]
			scales=scales,  # [N, 3]
			opacities=opacity.squeeze(-1),  # [N,]
			colors=shs,
			viewmats=viewmat[None],  # [1, 4, 4]
			Ks=K[None],  # [1, 3, 3]
			backgrounds=bg_color[None],
			width=int(viewpoint_camera.image_width),
			height=int(viewpoint_camera.image_height),
			packed=False,
			sh_degree=sh_degree,
			render_mode="RGB+D"
			)
			
			# process outputs
			render_RGBD = render_RGBD[0].permute(2, 0, 1) # [1, H, W, 4] -> [4, H, W]
			rendered_image = render_RGBD[:3]
			rendered_depth = render_RGBD[3:]
			rendered_alpha = rendered_alpha[0].permute(2, 0, 1)
			if len(info["radii"].shape) == 2:
				radii = info["radii"].squeeze(0) # [N,]
			elif len(info["radii"].shape) == 3:
				radii = info["radii"].squeeze(0).max(dim=-1).values
			# get normal maps
			batch_idx = kwargs["batch_idx"]
			rays_d = kwargs["rays_d"][batch_idx]
			rays_o = kwargs["rays_o"][batch_idx]
			xyz_map = rays_o + rendered_depth.permute(1, 2, 0) * rays_d
			normal_map = self.normal_module(xyz_map.permute(2, 0, 1).unsqueeze(0))[0]
			normal_map = F.normalize(normal_map, dim=0)
			normal_map = normal_map * 0.5 * rendered_alpha + 0.5
			
			# detach background pixels
			mask = rendered_alpha > 0.99
			normal_mask = mask.repeat(3, 1, 1)
			normal_map[~normal_mask] = normal_map[~normal_mask].detach()
			rendered_depth[~mask] = rendered_depth[~mask].detach()
			
			try:
				info["means2d"].retain_grad() # [1, N, 2]
			except:
				pass
			
			out["render"] = rendered_image.clamp(0, 1)
			out["depth"] = rendered_depth
			out["mask"] = rendered_alpha
			out["normal"] = normal_map
			out["viewspace_points"] = info["means2d"]
			out["visibility_filter"] = radii > 0
			out["radii"] = radii
		
		# render mask features
		if render_feature:
			feature_dim = pc.get_another_feature.shape[-1]
			feature_bg_color = bg_color[0].repeat(feature_dim)
			
			another_feature_map, _, _ = rasterization(
			means=means3D,  # [N, 3]
			quats=rotations,  # [N, 4]
			scales=scales,  # [N, 3]
			opacities=opacity.squeeze(-1),  # [N,]
			colors=pc.get_another_feature,
			viewmats=viewmat[None],  # [1, 4, 4]
			Ks=K[None],  # [1, 3, 3]
			backgrounds=feature_bg_color[None],
			width=int(viewpoint_camera.image_width),
			height=int(viewpoint_camera.image_height),
			packed=False,
			)
			another_feature_map = another_feature_map[0].permute(2, 0, 1)
			another_feature_map = F.normalize(another_feature_map, dim=0)
			out["another_feature"] = another_feature_map
		
		if render_instance:
			instance_masks = pc.get_instance_masks
			instancewise_rgbs = []
			instancewise_viewspace_points = []
			instancewise_visibility_filter = []
			instancewise_radii = []
			if render_instance_depth:
				instancewise_depths = []
			render_mode = "RGB+D" if render_instance_depth else "RGB"
			for i in range(instance_masks.shape[-1]):
				if len(instance_viewpoint_cams) > 0:
					viewmat = instance_viewpoint_cams[i].world_view_transform.transpose(0, 1).to(means3D.device) 
				m = instance_masks[:, i]
				instancewise_rgb, _, instancewise_info = rasterization(
				means=means3D[m],  # [N, 3]
				quats=rotations[m],  # [N, 4]
				scales=scales[m],  # [N, 3]
				opacities=opacity.squeeze(-1)[m],  # [N,]
				colors=shs[m],
				viewmats=viewmat[None],  # [1, 4, 4]
				Ks=K[None],  # [1, 3, 3]
				backgrounds=bg_color[None],
				width=int(viewpoint_camera.image_width),
				height=int(viewpoint_camera.image_height),
				packed=False,
				sh_degree=sh_degree,
				render_mode=render_mode
				)
				instancewise_rgb = instancewise_rgb[0].permute(2, 0, 1)
				if render_instance_depth:
					instancewise_depth = instancewise_rgb[3:]
					instancewise_rgb = instancewise_rgb[:3]
					instancewise_depths.append(instancewise_depth)
				instancewise_rgbs.append(instancewise_rgb)
				if len(instancewise_info["radii"].shape) == 2:
					instancewise_radii_ = instancewise_info["radii"].squeeze(0)
				elif len(instancewise_info["radii"].shape) == 3:
					instancewise_radii_ = instancewise_info["radii"].squeeze(0).max(dim=-1).values
				try:
					instancewise_info["means2d"].retain_grad() # [1, N, 2]
				except:
					pass
				instancewise_viewspace_points.append(instancewise_info["means2d"])
				instancewise_visibility_filter.append(instancewise_radii_ > 0)
				instancewise_radii.append(instancewise_radii_)
				
			instancewise_rgbs = torch.stack(instancewise_rgbs)
			out["instancewise_rgb"] = instancewise_rgbs
			out["instancewise_viewspace_points"] = instancewise_viewspace_points
			out["instancewise_visibility_filter"] = instancewise_visibility_filter
			out["instancewise_radii"] = instancewise_radii
			if render_instance_depth:
				instancewise_depths = torch.stack(instancewise_depths)
				out["instancewise_depth"] = instancewise_depths

		# Those Gaussians that were frustum culled or had a radius of 0 were not visible.
		# They will be excluded from value updates used in the splitting criteria.
		return out
