"""tomography_ml — machine-learning utilities for tomography localisation.

Catalog loading, lazy task datasets, localisation models, and training
helpers. Import submodules explicitly; the top-level namespace stays small.
"""

version = "0.0.1.dev0"
__version__ = version


from .device import get_device


__all__ = [
    "get_device",
]