import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.1"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )


from .data import multi_image, uncond
from .geometry import exporter, gaussian_base, gaussian_io
from .models.guidance import stable_diffusion_guidance_modified, stable_zero123_multiview_guidance
from .models.prompt_processors import stable_diffusion_multi_prompt_processor
from .renderer import gaussian_batch_renderer, gsplat_unified
from .system import dreamgrasp