from dataclasses import dataclass, field

import cv2
import numpy as np
import threestudio
import torch
from threestudio.models.background.base import BaseBackground
from threestudio.models.exporters.base import Exporter, ExporterOutput
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.mesh import Mesh
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("gaussian-mesh-exporter")
class MeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        fmt: str = "obj"
        save_name: str = "model"
        save_video: bool = True

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)

    def __call__(self) -> List[ExporterOutput]:
        mesh: Mesh = self.geometry.extract_mesh()
        return self.export_obj(mesh)

    def export_obj(self, mesh: Mesh) -> List[ExporterOutput]:
        params = {"mesh": mesh}
        return [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]

@threestudio.register("gaussian-instance-mesh-exporter")
class InstanceMeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        fmt: str = "obj"
        save_name: str = "model"
        save_video: bool = True

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)

    def __call__(self) -> List[ExporterOutput]:
        scene_mesh = self.geometry.extract_mesh()
        if self.geometry.instance_masks is not None:
            instance_meshes = self.geometry.extract_instance_mesh()
        else:
            instance_meshes = []
        return self.export_obj(scene_mesh, instance_meshes)

    def export_obj(self, scene_mesh, instance_meshes):
        params = {"mesh": scene_mesh}
        return_list = [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]
        for idx, instance_mesh in enumerate(instance_meshes):
            return_list += [
                ExporterOutput(
                    save_name=f"{self.cfg.save_name}_{idx}.obj", save_type="obj", params={"mesh": instance_mesh}
                )
            ]
        return return_list