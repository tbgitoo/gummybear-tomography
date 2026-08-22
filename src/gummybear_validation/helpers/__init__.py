"""Shared accessors and notebook-style assertion helpers for validation code.

These utilities normalize dataclass / dict / attribute objects and print
PASS/FAIL diagnostics used across particle-transport and sequence checks.
Pinhole camera helpers align validation plots with ``gummybear.rays.camera``.
"""

from .access_helpers import (
    as_mapping,
    check_close,
    check_true,
    event_entry_t,
    event_exit_t,
    event_segment_index,
    get_any,
    get_field,
    pair_path_id,
    print_object_inventory,
    require_any,
    summarize_array,
)
from .camera_helpers import (
    pinhole_camera_basis,
    pinhole_camera_framing_mesh,
    trimesh_camera_transform,
)
from .m2b_debug import M2BDebugProxyResult, compute_m2b_debug_proxy, print_m2b_face_fields

__all__ = [
    "as_mapping",
    "get_any",
    "require_any",
    "check_true",
    "check_close",
    "summarize_array",
    "print_object_inventory",
    "get_field",
    "pair_path_id",
    "event_segment_index",
    "event_exit_t",
    "event_entry_t",
    "pinhole_camera_basis",
    "pinhole_camera_framing_mesh",
    "trimesh_camera_transform",
    "M2BDebugProxyResult",
    "compute_m2b_debug_proxy",
    "print_m2b_face_fields",
]
