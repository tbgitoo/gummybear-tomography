"""POV-Ray illustration helpers. Separate from optical simulation / generation."""

from .export_m8_network_scene import export_m8_network_scene
from .export_m8_physical_scene import export_m8_physical_scene, render_pov_file
from .figure3_export import export_figure3_convergence
from .load_sample import PhysicalSetup, load_m8_physical_setup
from .pov_scene import IllustrationCameraParams

version = "0.0.1.dev0"
__version__ = version

__all__ = [
    "IllustrationCameraParams",
    "PhysicalSetup",
    "export_figure3_convergence",
    "export_m8_network_scene",
    "export_m8_physical_scene",
    "load_m8_physical_setup",
    "render_pov_file",
    "version",
]
