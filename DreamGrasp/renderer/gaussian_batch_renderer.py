import torch
from threestudio.utils.ops import get_cam_info_gaussian
from torch.cuda.amp import autocast

from ..geometry.gaussian_base import BasicPointCloud, Camera


class GaussianBatchRenderer:
    def batch_forward(self, batch):
        bs = batch["c2w"].shape[0]
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        pred_normals = []
        depths = []
        masks = []
        another_features = []
        instancewise_rgbs = []
        instancewise_viewspace_points = []
        instancewise_visibility_filter = []
        instancewise_radii = []
        instancewise_depths = []
        
        for batch_idx in range(bs):
            batch["batch_idx"] = batch_idx
            fovy = batch["fovy"][batch_idx]
            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=batch["c2w"][batch_idx], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
            )
            viewpoint_cam = Camera(
                FoVx=fovy,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )
            
            if batch.get("render_instance", False):
                instance_viewpoint_cams = []
                if "instancewise_c2w" in batch:
                    for instance_idx in range(batch["instancewise_c2w"].shape[1]):
                        w2c, proj, cam_p = get_cam_info_gaussian(
                            c2w=batch["instancewise_c2w"][batch_idx][instance_idx], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
                        )
                        instance_viewpoint_cams.append(
                            Camera(
                                FoVx=fovy,
                                FoVy=fovy,
                                image_width=batch["width"],
                                image_height=batch["height"],
                                world_view_transform=w2c,
                                full_proj_transform=proj,
                                camera_center=cam_p,
                            )
                        )
                batch["instance_viewpoint_cams"] = instance_viewpoint_cams
            # import pdb; pdb.set_trace()
            with autocast(enabled=False):
                render_pkg = self.forward(
                    viewpoint_cam, self.background_tensor, **batch
                )
                if render_pkg.__contains__("render"):
                    renders.append(render_pkg["render"])
                if render_pkg.__contains__("viewspace_points"):
                    viewspace_points.append(render_pkg["viewspace_points"])
                if render_pkg.__contains__("visibility_filter"):
                    visibility_filters.append(render_pkg["visibility_filter"])
                if render_pkg.__contains__("radii"):
                    radiis.append(render_pkg["radii"])
                if render_pkg.__contains__("normal"):
                    normals.append(render_pkg["normal"])
                if (
                    render_pkg.__contains__("pred_normal")
                    and render_pkg["pred_normal"] is not None
                ):
                    pred_normals.append(render_pkg["pred_normal"])
                if render_pkg.__contains__("depth"):
                    depths.append(render_pkg["depth"])
                if render_pkg.__contains__("mask"):
                    masks.append(render_pkg["mask"])
                if render_pkg.__contains__("another_feature"):
                    another_features.append(render_pkg["another_feature"])
                if render_pkg.__contains__("instancewise_rgb"):
                    instancewise_rgbs.append(render_pkg["instancewise_rgb"])
                if render_pkg.__contains__("instancewise_viewspace_points"):
                    instancewise_viewspace_points.append(render_pkg["instancewise_viewspace_points"])
                if render_pkg.__contains__("instancewise_visibility_filter"):
                    instancewise_visibility_filter.append(render_pkg["instancewise_visibility_filter"])
                if render_pkg.__contains__("instancewise_radii"):
                    instancewise_radii.append(render_pkg["instancewise_radii"])
                if render_pkg.__contains__("instancewise_depth"):
                    instancewise_depths.append(render_pkg["instancewise_depth"])

        outputs = {}
        if len(renders) > 0:
            outputs.update({"comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1)})
        if len(viewspace_points) > 0:
            outputs.update({"viewspace_points": viewspace_points})
        if len(visibility_filters) > 0:
            outputs.update({"visibility_filter": visibility_filters})
        if len(radiis) > 0:
            outputs.update({"radii": radiis})
        if len(normals) > 0:
            outputs.update(
                {
                    "comp_normal": torch.stack(normals, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(pred_normals) > 0:
            outputs.update(
                {
                    "comp_pred_normal": torch.stack(pred_normals, dim=0).permute(
                        0, 2, 3, 1
                    ),
                }
            )
        if len(depths) > 0:
            outputs.update(
                {
                    "comp_depth": torch.stack(depths, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(masks) > 0:
            outputs.update(
                {
                    "comp_mask": torch.stack(masks, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(another_features) > 0:
            outputs.update(
                {
                    "comp_another_feature": torch.stack(another_features, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(instancewise_rgbs) > 0:
            outputs.update(
                {
                    "comp_instancewise_rgb": torch.stack(instancewise_rgbs, dim=0).permute(0, 1, 3, 4, 2),
                }
            )
        if len(instancewise_viewspace_points) > 0:
            outputs.update({"instancewise_viewspace_points": instancewise_viewspace_points})
        if len(instancewise_visibility_filter) > 0:
            outputs.update({"instancewise_visibility_filter": instancewise_visibility_filter})
        if len(instancewise_radii) > 0:
            outputs.update({"instancewise_radii": instancewise_radii})
        if len(instancewise_depths) > 0:
            outputs.update(
                {
                    "comp_instancewise_depth": torch.stack(instancewise_depths, dim=0).permute(0, 1, 3, 4, 2),
                }
            )
            
        return outputs
