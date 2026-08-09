"""Device selection for PyTorch training and inference.

Host accelerator handling (CUDA / Apple MPS / CPU).
"""

version = "0.0.1.dev0"
__version__ = version

from .device import get_device


__all__ = [
    "get_device",
]

