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
from .m3_transport import (
    assert_source_hit_contract,
    assert_source_ray_bundle_invariants,
    print_entry_refraction_summary,
    print_face_state_summary,
    print_source_hit_summary,
)
from .m4_diffusion import (
    assert_deposition_conservation,
    assert_deposition_sanity,
    assert_live_netgen_mesh,
    assert_m4e_hybrid_checks,
    build_diffusion_ray_subset,
    make_centroid_axis_ray,
    m4e_metadata_template,
    print_deposition_summary,
    print_diffusion_mesh_summary,
    print_m4e_metadata,
    write_m4e_artifacts,
)

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
    "assert_source_ray_bundle_invariants",
    "print_source_hit_summary",
    "assert_source_hit_contract",
    "print_entry_refraction_summary",
    "print_face_state_summary",
    "assert_live_netgen_mesh",
    "print_diffusion_mesh_summary",
    "make_centroid_axis_ray",
    "print_deposition_summary",
    "assert_deposition_conservation",
    "assert_deposition_sanity",
    "build_diffusion_ray_subset",
    "write_m4e_artifacts",
    "print_m4e_metadata",
    "m4e_metadata_template",
    "assert_m4e_hybrid_checks",
]
