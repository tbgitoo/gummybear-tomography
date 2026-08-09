"""Analytic particle geometry and ray/source perturbation helpers.

Particles are continuous-space spherical inclusions. They are not meshed and
do not change the diffusion mesh or operator; they only perturb ray transport
and volumetric source terms.
"""

from .geometry import (
    DEFAULT_OVERLAP_GAP_TOL,
    DEFAULT_PATH_LENGTH_TOL,
    ParticleIntersectionEvent,
    ParticleOverlapError,
    ParticleSet,
    ParticleSphere,
    intersect_segments_with_particles,
    segment_sphere_intersection,
)
from .placement import sample_random_centers_in_mesh
from .perturbation import (
    PARTITION_MODEL,
    AffectedTransportPair,
    AffectedTransportPairResult,
    ParticleScatterDepositionResult,
    ParticleScatterSourceEvent,
    TransportInterval,
    TransportPairDepositionResult,
    TransportSourceCorrectionResult,
    assert_downstream_background_shadow,
    build_affected_transport_pairs,
    compute_transport_source_correction,
    deposit_particle_scatter_sources,
    deposit_transport_pair_sources,
    find_containing_tet,
    nearest_tet_centroid,
    partition_particle_loss,
    point_in_tetrahedron,
)
from .access_helpers import (as_mapping, get_any, 
    require_any,
    check_true,
    check_close,
    summarize_array,
    print_object_inventory,
    get_field,
    pair_path_id,
    event_segment_index,
    event_exit_t


    

    )

__all__ = [
    "DEFAULT_OVERLAP_GAP_TOL",
    "DEFAULT_PATH_LENGTH_TOL",
    "ParticleIntersectionEvent",
    "ParticleOverlapError",
    "ParticleSet",
    "ParticleSphere",
    "intersect_segments_with_particles",
    "segment_sphere_intersection",
    "sample_random_centers_in_mesh",
    "PARTITION_MODEL",
    "AffectedTransportPair",
    "AffectedTransportPairResult",
    "ParticleScatterDepositionResult",
    "ParticleScatterSourceEvent",
    "TransportInterval",
    "TransportPairDepositionResult",
    "TransportSourceCorrectionResult",
    "assert_downstream_background_shadow",
    "build_affected_transport_pairs",
    "compute_transport_source_correction",
    "deposit_particle_scatter_sources",
    "deposit_transport_pair_sources",
    "find_containing_tet",
    "nearest_tet_centroid",
    "partition_particle_loss",
    "point_in_tetrahedron",
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
    "event_exit_t"
]

