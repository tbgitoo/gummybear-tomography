"""Translucent camera forward-model optics for GummyBearTomography.

Covers face illumination and transmittance proxies, Snell refraction and
face-level direct transport, coarse tetrahedral diffusion meshing and finite-element method (FEM)
solve, volumetric source deposition, diffuse fluence field ``Phi`` sampling at camera hits,
and hybrid direct-plus-diffuse image composition.

Import from here for the stable public API; submodules hold the implementations.
"""

from .camera_light_collection import beam_view_coupling, beam_vector_field_to_camera_image
from .diffuse_sampling import (
    DiffuseSampleResult,
    PhiSamplingLocalization,
    apply_phi_localization,
    interpolate_phi_nodes_to_points,
    localize_points_in_diffusion_mesh,
    sample_diffuse_image,
    sample_phi_at_hit_points,
)
from .diffusion_mesh import DiffusionMesh, DiffusionMeshMetadata, generate_diffusion_mesh
from .diffusion_solve import (
    ROBIN_BOUNDARY_MODEL,
    DiffusionSolveResult,
    default_diffusion_coefficient,
    operator_cache_keys,
    sample_phi_at_points,
    solve_diffusion,
)
from .face_illumination import sample_face_values_to_image
from .face_transport import (
    FaceOpticalState,
    RefractiveDirectImageResult,
    accumulate_source_coverage,
    compute_refractive_direct_image,
    compute_refracted_face_field,
    propagate_entry_exit_transport,
    propagate_source_rays,
    refractive_exit_view_coupling,
    sample_face_state_to_camera,
)
from .hybrid_compose import FORWARD_MODEL_TIER, HybridImageResult, compose_hybrid_image
from .illumination_pass import (
    CameraSampledBeamVectorField,
    camera_sampled_beam_vector_field,
    source_intensity_at_faces,
)
from .light_path import compute_face_upstream_thickness, thickness_to_transmittance
from .light_source import (
    DirectionalLightConfig,
    LightConfig,
    PointLightConfig,
    illumination_directions_at_faces,
)
from .material import OpticalMaterialConfig
from .refraction import (
    RefractedRayBundleResult,
    refract_direction,
    refract_directions,
    refract_ray_bundle,
)
from .source_deposition import (
    RaySegmentBundle,
    SourceDepositionResult,
    deposit_ray_source,
    distribute_energy_along_segment,
    in_object_segments_from_rays,
    make_synthetic_axis_ray,
)
from .source_sampling import (
    SourceSamplingParams,
    generate_directional_source_rays,
    generate_point_source_rays,
    make_source_ray_bundle,
)


__all__ = [
    "sample_face_values_to_image",
    "PointLightConfig",
    "illumination_directions_at_faces",
    "DirectionalLightConfig",
    "LightConfig",
    "compute_face_upstream_thickness",
    "thickness_to_transmittance",
    "source_intensity_at_faces",
    "CameraSampledBeamVectorField",
    "beam_view_coupling",
    "beam_vector_field_to_camera_image",
    "camera_sampled_beam_vector_field",
    "OpticalMaterialConfig",
    "SourceSamplingParams",
    "make_source_ray_bundle",
    "generate_directional_source_rays",
    "generate_point_source_rays",
    "refract_direction",
    "refract_directions",
    "refract_ray_bundle",
    "RefractedRayBundleResult",
    "FaceOpticalState",
    "RefractiveDirectImageResult",
    "accumulate_source_coverage",
    "compute_refractive_direct_image",
    "propagate_entry_exit_transport",
    "propagate_source_rays",
    "compute_refracted_face_field",
    "refractive_exit_view_coupling",
    "sample_face_state_to_camera",
    # M4
    "DiffusionMesh",
    "DiffusionMeshMetadata",
    "generate_diffusion_mesh",
    "RaySegmentBundle",
    "SourceDepositionResult",
    "deposit_ray_source",
    "distribute_energy_along_segment",
    "in_object_segments_from_rays",
    "make_synthetic_axis_ray",
    "ROBIN_BOUNDARY_MODEL",
    "DiffusionSolveResult",
    "default_diffusion_coefficient",
    "operator_cache_keys",
    "solve_diffusion",
    "sample_phi_at_points",
    "DiffuseSampleResult",
    "PhiSamplingLocalization",
    "apply_phi_localization",
    "interpolate_phi_nodes_to_points",
    "localize_points_in_diffusion_mesh",
    "sample_phi_at_hit_points",
    "sample_diffuse_image",
    "FORWARD_MODEL_TIER",
    "HybridImageResult",
    "compose_hybrid_image",
]
