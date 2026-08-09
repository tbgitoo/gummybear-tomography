"""Focused M5A-M5C tests; no FEM dependency required."""

from __future__ import annotations

import numpy as np
import pytest

from gummybear.optics.diffusion_mesh import DiffusionMesh, DiffusionMeshMetadata
from gummybear.optics.source_deposition import (
    RaySegmentBundle,
    deposit_ray_source,
    make_synthetic_axis_ray,
)
from gummybear.particles import (
    ParticleSet,
    ParticleSphere,
    build_affected_transport_pairs,
    compute_transport_source_correction,
    deposit_particle_scatter_sources,
    intersect_segments_with_particles,
    nearest_tet_centroid,
    partition_particle_loss,
    segment_sphere_intersection,
)

from gummybear.particles.access_helpers import (
    get_any,
    require_any,
    pair_path_id,
    event_segment_index,
    event_entry_t,
    event_exit_t,
)

from gummybear_validation.text_output import array_mini_summary


def _single_tet_mesh() -> DiffusionMesh:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return DiffusionMesh(
        nodes=nodes,
        tets=np.array([[0, 1, 2, 3]], dtype=int),
        centroids=nodes.mean(axis=0, keepdims=True),
        volumes=np.array([1.0 / 6.0], dtype=float),
        metadata=DiffusionMeshMetadata(
            stl_hash="test",
            geometry_id="single_tet",
            meshing_method="synthetic",
            num_elements=1,
            num_nodes=4,
        ),
        netgen_mesh=None,
    )


def _event_energies(result) -> dict[int, float]:
    return {
        event.particle_index: event.E_loss
        for pair in result.pairs
        for event in pair.particle_scatter_source_events
    }


# M5A geometry


def test_centerline_hit_path_length_2r():
    event = segment_sphere_intersection(
        np.array([-2.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.zeros(3),
        0.5,
    )
    assert event is not None
    assert event.path_length_inside_particle == pytest.approx(1.0)


def test_miss_returns_none():
    event = segment_sphere_intersection(
        np.array([-2.0, 3.0, 0.0]),
        np.array([2.0, 3.0, 0.0]),
        np.zeros(3),
        1.0,
    )
    assert event is None


def test_tangent_hit_is_ignored():
    event = segment_sphere_intersection(
        np.array([-2.0, 1.0, 0.0]),
        np.array([2.0, 1.0, 0.0]),
        np.zeros(3),
        1.0,
        path_length_tol=1e-9,
    )
    assert event is None


def test_segment_starting_inside_sphere_is_clipped():
    event = segment_sphere_intersection(
        np.zeros(3),
        np.array([3.0, 0.0, 0.0]),
        np.zeros(3),
        1.0,
    )
    assert event is not None
    assert event.entry_t == pytest.approx(0.0)
    assert event.path_length_inside_particle == pytest.approx(1.0)


def test_multiple_particle_events_are_sorted():
    particles = ParticleSet.from_particles(
        [
            ParticleSphere(center=(2.0, 0.0, 0.0), radius=0.25),
            ParticleSphere(center=(0.0, 0.0, 0.0), radius=0.25),
        ]
    )
    events = intersect_segments_with_particles(
        np.array([[-2.0, 0.0, 0.0]]),
        np.array([[4.0, 0.0, 0.0]]),
        particles,
    )
    assert [event.particle_index for event in events] == [1, 0]
    assert events[0].entry_t < events[1].entry_t


# M5B clean/dirty transport pairs


def test_partition_particle_loss_uses_relative_mu():
    E_abs, E_scat = partition_particle_loss(10.0, mu_abs=2.0, mu_scat=8.0)
    assert E_abs == pytest.approx(2.0)
    assert E_scat == pytest.approx(8.0)


def test_no_particle_produces_no_affected_pair():
    segments = make_synthetic_axis_ray((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    particles = [ParticleSphere(center=(0.0, 2.0, 0.0), radius=0.1)]
    result = build_affected_transport_pairs(segments, particles)
    assert result.pairs == ()
    assert result.affected_path_ids == ()
    assert result.affected_segment_indices.size == 0


def test_single_pure_absorber_lowers_dirty_intensity_without_scatter_source():
    segments = make_synthetic_axis_ray((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    particles = [ParticleSphere(center=(0.0, 0.0, 0.0), radius=0.5, mu_abs=2.0)]
    result = build_affected_transport_pairs(segments, particles)
    pair = result.pairs[0]
    assert pair.dirty_output_intensity < pair.clean_output_intensity
    assert pair.total_E_abs > 0.0
    assert pair.total_E_scat == pytest.approx(0.0)
    assert sum(event.E_scat for event in pair.particle_scatter_source_events) == 0.0


def test_single_pure_scatterer_generates_positive_scatter_source():
    segments = make_synthetic_axis_ray((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    particles = [
        ParticleSphere(center=(0.0, 0.0, 0.0), radius=0.5, mu_scat=2.0)
    ]
    pair = build_affected_transport_pairs(segments, particles).pairs[0]
    assert pair.dirty_output_intensity < pair.clean_output_intensity
    assert pair.total_E_abs == pytest.approx(0.0)
    assert pair.total_E_scat > 0.0
    assert sum(event.E_scat for event in pair.particle_scatter_source_events) > 0.0


def test_mixed_particle_energy_partition_closes():
    segments = make_synthetic_axis_ray((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    particles = [
        ParticleSphere(
            center=(0.0, 0.0, 0.0),
            radius=0.5,
            mu_abs=1.0,
            mu_scat=3.0,
        )
    ]
    pair = build_affected_transport_pairs(segments, particles).pairs[0]
    event = pair.particle_scatter_source_events[0]
    assert event.E_abs + event.E_scat == pytest.approx(
        event.I_before - event.I_after
    )
    assert event.E_scat / event.E_abs == pytest.approx(3.0)


def test_dirty_path_applies_background_attenuation_before_particle():
    segments = make_synthetic_axis_ray((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    particles = [
        ParticleSphere(center=(0.0, 0.0, 0.0), radius=0.5, mu_scat=2.0)
    ]
    without_bg = build_affected_transport_pairs(segments, particles).pairs[0]
    with_bg = build_affected_transport_pairs(
        segments, particles, mu_s=1.0, mu_a=0.0
    ).pairs[0]
    event_bg = with_bg.particle_scatter_source_events[0]
    event_no = without_bg.particle_scatter_source_events[0]
    assert event_bg.I_before < event_no.I_before
    particle_iv = next(
        interval
        for interval in with_bg.dirty_intervals
        if interval.active_particle_indices
    )
    approach = next(
        interval
        for interval in with_bg.dirty_intervals
        if not interval.active_particle_indices
        and interval.interval_id < particle_iv.interval_id
    )
    assert approach.I_out < approach.I_in
    assert particle_iv.I_in == pytest.approx(approach.I_out)
    after = next(
        interval
        for interval in with_bg.dirty_intervals
        if not interval.active_particle_indices
        and interval.interval_id > particle_iv.interval_id
    )
    assert after.I_in == pytest.approx(particle_iv.I_out)
    assert after.I_out < after.I_in


def test_downstream_dirty_background_does_not_exceed_clean_when_particle_is_thicker():
    from gummybear.particles import assert_downstream_background_shadow

    mesh = _single_tet_mesh()
    c = mesh.centroids[0]
    # Long approach before a particle denser than the background medium.
    segments = RaySegmentBundle(
        starts=np.array([[-1.0, c[1], c[2]]]),
        ends=np.array([[1.5, c[1], c[2]]]),
        intensities=np.array([1.0]),
        ray_ids=np.array([8]),
    )
    particles = [
        ParticleSphere(center=(c[0] + 0.35, c[1], c[2]), radius=0.08, mu_scat=2.0)
    ]
    pairs = build_affected_transport_pairs(
        segments, particles, mu_s=0.3, mu_a=0.1
    )
    summary = assert_downstream_background_shadow(
        mesh, pairs.pairs[0], mu_s=0.3, mu_a=0.1, density=True
    )
    assert summary["n_violations"] == 0


def test_two_particles_receive_ordered_dirty_intensity():
    segments = make_synthetic_axis_ray((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    particles = [
        ParticleSphere(center=(-1.0, 0.0, 0.0), radius=0.25, mu_abs=1.0),
        ParticleSphere(center=(1.0, 0.0, 0.0), radius=0.25, mu_scat=2.0),
    ]
    events = build_affected_transport_pairs(
        segments, particles
    ).pairs[0].particle_scatter_source_events
    assert [event.particle_index for event in events] == [0, 1]
    assert events[1].I_before == pytest.approx(events[0].I_after)


def test_reversing_path_changes_per_particle_energy_bookkeeping():
    particles = [
        ParticleSphere(center=(-1.0, 0.0, 0.0), radius=0.25, mu_abs=0.5),
        ParticleSphere(center=(1.0, 0.0, 0.0), radius=0.25, mu_scat=3.0),
    ]
    forward = build_affected_transport_pairs(
        make_synthetic_axis_ray((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        particles,
    )
    reverse = build_affected_transport_pairs(
        make_synthetic_axis_ray((3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)),
        particles,
    )
    assert _event_energies(forward)[0] != pytest.approx(_event_energies(reverse)[0])
    assert _event_energies(forward)[1] != pytest.approx(_event_energies(reverse)[1])


def test_overlapping_particle_intervals_fail_loudly_when_geometry_check_bypassed():
    """Interval-overlap guard remains for invalid sets built with bypass."""
    segments = make_synthetic_axis_ray((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    particles = ParticleSet.from_particles(
        [
            ParticleSphere(center=(-0.2, 0.0, 0.0), radius=0.5, mu_abs=1.0),
            ParticleSphere(center=(0.2, 0.0, 0.0), radius=0.5, mu_scat=1.0),
        ],
        require_non_overlapping=False,
    )
    with pytest.raises(ValueError, match="Overlapping particle intervals"):
        build_affected_transport_pairs(segments, particles)


def test_geometrically_overlapping_particles_rejected_by_default():
    from gummybear.particles import ParticleOverlapError

    with pytest.raises(ParticleOverlapError, match="overlapping spheres"):
        ParticleSet.from_particles(
            [
                ParticleSphere(center=(-0.2, 0.0, 0.0), radius=0.5),
                ParticleSphere(center=(0.2, 0.0, 0.0), radius=0.5),
            ]
        )


def test_particle_set_validate_allows_touching_and_separated_spheres():
    touching = ParticleSet.from_particles(
        [
            ParticleSphere(center=(-1.0, 0.0, 0.0), radius=0.5, particle_id="a"),
            ParticleSphere(center=(1.0, 0.0, 0.0), radius=0.5, particle_id="b"),
        ]
    )
    assert touching.validate() is True
    touching.require_valid()

    separated = ParticleSet.from_particles(
        [
            ParticleSphere(center=(-2.0, 0.0, 0.0), radius=0.5),
            ParticleSphere(center=(2.0, 0.0, 0.0), radius=0.5),
        ]
    )
    assert separated.validate() is True

    overlapping = ParticleSet(
        particles=(
            ParticleSphere(center=(0.0, 0.0, 0.0), radius=1.0),
            ParticleSphere(center=(1.0, 0.0, 0.0), radius=1.0),
        )
    )
    assert overlapping.validate() is False


def test_segment_rows_and_transport_path_identity_remain_distinct():
    segments = RaySegmentBundle(
        starts=np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ends=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        intensities=np.array([1.0, 1.0]),
        ray_ids=np.array([42, 42]),
        path_order=np.array([0, 1]),
    )
    particles = [
        ParticleSphere(center=(1.0, 0.0, 0.0), radius=0.25, mu_abs=2.0)
    ]
    result = build_affected_transport_pairs(segments, particles)
    assert result.affected_path_ids == (42,)
    assert result.affected_segment_indices.tolist() == [1]
    assert result.affected_path_ids != tuple(result.affected_segment_indices)
    assert len(result.pairs[0].clean_intervals) == 2


def test_path_order_controls_multi_segment_particle_sequence():
    segments = RaySegmentBundle(
        starts=np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
        ends=np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        intensities=np.array([1.0, 1.0]),
        ray_ids=np.array([7, 7]),
        path_order=np.array([1, 0]),
    )
    particles = [
        ParticleSphere(center=(-1.0, 0.0, 0.0), radius=0.2, mu_abs=1.0),
        ParticleSphere(center=(1.0, 0.0, 0.0), radius=0.2, mu_scat=1.0),
    ]
    events = build_affected_transport_pairs(
        segments, particles
    ).pairs[0].particle_scatter_source_events
    assert [event.segment_index for event in events] == [1, 0]
    assert events[1].I_before == pytest.approx(events[0].I_after)


# M5C source correction from pairs


def test_particle_scatter_deposition_uses_events_owned_by_pairs():
    mesh = _single_tet_mesh()
    c = mesh.centroids[0]
    segments = RaySegmentBundle(
        starts=np.array([[0.05, c[1], c[2]]]),
        ends=np.array([[0.45, c[1], c[2]]]),
        intensities=np.array([1.0]),
        ray_ids=np.array([9]),
    )
    particles = [ParticleSphere(center=c, radius=0.1, mu_scat=2.0)]
    pairs = build_affected_transport_pairs(
        segments, particles, mu_s=1.0, mu_a=0.0
    )
    deposition = deposit_particle_scatter_sources(mesh, pairs)
    assert deposition.assignment_mode == "attenuated_chord"
    assert deposition.metadata["distribution"] == "exact_ray_tet_beer_lambert"
    assert deposition.total_E_scat_deposited == pytest.approx(pairs.total_E_scat)
    assert np.sum(deposition.delta_E_particle_scat_elem) > 0.0
    event = pairs.pairs[0].particle_scatter_source_events[0]
    assert event.E_scat > 0.0
    assert np.allclose(event.point, 0.5 * (event.entry_point + event.exit_point))


def _two_tet_mesh() -> DiffusionMesh:
    """Two tets sharing the face (0,1,2); chord along +z crosses both."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 1.0],
            [0.25, 0.25, -1.0],
        ],
        dtype=float,
    )
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=int)
    centroids = np.array([nodes[tet].mean(axis=0) for tet in tets], dtype=float)
    volumes = np.array(
        [
            abs(np.linalg.det(np.column_stack((nodes[1] - nodes[0], nodes[2] - nodes[0], nodes[3] - nodes[0])))) / 6.0,
            abs(np.linalg.det(np.column_stack((nodes[1] - nodes[0], nodes[2] - nodes[0], nodes[4] - nodes[0])))) / 6.0,
        ],
        dtype=float,
    )
    return DiffusionMesh(
        nodes=nodes,
        tets=tets,
        centroids=centroids,
        volumes=volumes,
        metadata=DiffusionMeshMetadata(
            stl_hash="test",
            geometry_id="two_tet",
            meshing_method="synthetic",
            num_elements=2,
            num_nodes=5,
        ),
        netgen_mesh=None,
    )


def test_particle_scatter_deposition_follows_beer_lambert_along_chord():
    from gummybear.particles.perturbation import (
        AffectedTransportPair,
        AffectedTransportPairResult,
        ParticleScatterSourceEvent,
    )

    mesh = _two_tet_mesh()
    entry = np.array([0.25, 0.25, -0.5])
    exit_ = np.array([0.25, 0.25, 0.5])
    midpoint = 0.5 * (entry + exit_)
    mu_scat = 2.0
    mu_abs = 0.0
    I_before = 1.0
    length = float(np.linalg.norm(exit_ - entry))
    I_after = I_before * np.exp(-(mu_scat + mu_abs) * length)
    E_scat = I_before - I_after
    event = ParticleScatterSourceEvent(
        path_id=1,
        segment_index=0,
        interval_id=0,
        particle_index=0,
        entry_point=entry,
        exit_point=exit_,
        point=midpoint,
        E_scat=E_scat,
        E_abs=0.0,
        I_before=I_before,
        I_after=I_after,
        metadata={"mu_abs": mu_abs, "mu_scat": mu_scat, "path_length": length},
    )
    pair = AffectedTransportPair(
        path_id=1,
        clean_intervals=(),
        dirty_intervals=(),
        particle_events=(),
        particle_scatter_source_events=(event,),
        clean_output_intensity=1.0,
        dirty_output_intensity=I_after,
        total_E_abs=0.0,
        total_E_scat=E_scat,
    )
    pairs = AffectedTransportPairResult(
        pairs=(pair,),
        affected_path_ids=(1,),
        affected_segment_indices=np.array([0], dtype=int),
        total_E_abs=0.0,
        total_E_scat=E_scat,
    )
    deposition = deposit_particle_scatter_sources(mesh, pairs)
    assert deposition.assignment_mode == "attenuated_chord"
    assert deposition.metadata["distribution"] == "exact_ray_tet_beer_lambert"
    assert deposition.total_E_scat_deposited == pytest.approx(E_scat)
    # Entry side is tet 1 (z < 0); exit side is tet 0 (z > 0).
    half = 0.5
    expected_entry = I_before * (1.0 - np.exp(-mu_scat * half))
    expected_exit = I_before * np.exp(-mu_scat * half) * (1.0 - np.exp(-mu_scat * half))
    assert deposition.delta_E_particle_scat_elem[1] == pytest.approx(expected_entry)
    assert deposition.delta_E_particle_scat_elem[0] == pytest.approx(expected_exit)
    assert deposition.delta_E_particle_scat_elem[1] > deposition.delta_E_particle_scat_elem[0]

    uniform = deposit_particle_scatter_sources(
        mesh, pairs, assignment="chord_length"
    )
    assert uniform.delta_E_particle_scat_elem[0] == pytest.approx(0.5 * E_scat)
    assert uniform.delta_E_particle_scat_elem[1] == pytest.approx(0.5 * E_scat)

    midpoint_only = deposit_particle_scatter_sources(
        mesh, pairs, assignment="containing_tet"
    )
    assert midpoint_only.metadata["distribution"] == "midpoint_point_assignment"
    assert np.count_nonzero(midpoint_only.delta_E_particle_scat_elem > 0.0) == 1
    assert float(np.sum(midpoint_only.delta_E_particle_scat_elem)) == pytest.approx(E_scat)


def test_source_correction_identity_and_inspectable_split():
    mesh = _single_tet_mesh()
    c = mesh.centroids[0]
    segments = RaySegmentBundle(
        starts=np.array([[-0.5, c[1], c[2]]]),
        ends=np.array([[1.5, c[1], c[2]]]),
        intensities=np.array([1.0]),
        ray_ids=np.array([11]),
    )
    particles = [
        ParticleSphere(center=c, radius=0.1, mu_abs=1.0, mu_scat=2.0)
    ]
    pairs = build_affected_transport_pairs(
        segments, particles, mu_s=1.0, mu_a=0.0
    )
    clean = deposit_ray_source(mesh, segments, mu_s=1.0, mu_a=0.0)
    source = compute_transport_source_correction(
        mesh,
        pairs,
        clean.E_scat_elem,
        mu_s=1.0,
        mu_a=0.0,
    )
    assert np.allclose(
        source.delta_E_transport_elem,
        source.delta_E_background_elem + source.delta_E_particle_scat_elem,
    )
    assert np.allclose(
        source.E_particle_elem,
        source.E_clean_elem + source.delta_E_transport_elem,
    )
    assert np.allclose(source.S_particle, source.E_particle_elem / mesh.volumes)
    assert source.metadata["source_model"] == "affected_transport_pair_delta"


def test_no_affected_pairs_produce_zero_source_delta():
    mesh = _single_tet_mesh()
    segments = RaySegmentBundle(
        starts=np.array([[0.1, 0.1, 0.1]]),
        ends=np.array([[0.2, 0.1, 0.1]]),
        intensities=np.array([1.0]),
        ray_ids=np.array([3]),
    )
    pairs = build_affected_transport_pairs(
        segments,
        [ParticleSphere(center=(2.0, 2.0, 2.0), radius=0.1, mu_abs=1.0)],
    )
    clean = np.array([0.25])
    source = compute_transport_source_correction(
        mesh, pairs, clean, mu_s=1.0, mu_a=0.0
    )
    assert np.allclose(source.delta_E_transport_elem, 0.0)
    assert np.allclose(source.E_particle_elem, clean)


def test_pure_absorber_does_not_enter_particle_scatter_source():
    mesh = _single_tet_mesh()
    c = mesh.centroids[0]
    segments = RaySegmentBundle(
        starts=np.array([[0.05, c[1], c[2]]]),
        ends=np.array([[0.45, c[1], c[2]]]),
        intensities=np.array([1.0]),
        ray_ids=np.array([5]),
    )
    pairs = build_affected_transport_pairs(
        segments,
        [ParticleSphere(center=c, radius=0.1, mu_abs=5.0)],
        mu_s=1.0,
        mu_a=0.0,
    )
    clean = deposit_ray_source(mesh, segments, mu_s=1.0, mu_a=0.0)
    source = compute_transport_source_correction(
        mesh, pairs, clean.E_scat_elem, mu_s=1.0, mu_a=0.0
    )
    assert np.allclose(source.delta_E_particle_scat_elem, 0.0)


def test_nearest_tet_centroid_is_deterministic():
    mesh = _single_tet_mesh()
    assert nearest_tet_centroid(mesh, np.array([10.0, 10.0, 10.0])) == 0


def test_particle_manifest_preserves_fixed_mesh_contract():
    particle_set = ParticleSet.from_particles(
        [ParticleSphere(center=(0, 0, 0), radius=0.5, particle_id="p000")]
    )
    block = particle_set.to_manifest_block()
    assert block["requires_remeshing"] is False
    assert block["changes_diffusion_operator"] is False


def test_ray_segment_bundle_transport_lineage_contract():
    segments = RaySegmentBundle(
        starts=np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        ends=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        intensities=np.ones(4, dtype=float),
        ray_ids=np.array([42, 42, 99, 99], dtype=int),
        path_order=np.array([0, 1, 0, 1], dtype=int),
    )

    ray_ids = get_any(
        segments,
        ["path_ids", "ray_ids", "transport_path_ids"],
        default=None,
    )

    segment_ids = get_any(
        segments,
        ["segment_ids"],
        default=None,
    )

    path_order = get_any(
        segments,
        ["path_order", "segment_order"],
        default=None,
    )

    assert hasattr(segments, "starts")
    assert hasattr(segments, "ends")
    assert hasattr(segments, "intensities")

    assert ray_ids is not None, (
        "segments must expose path_ids / ray_ids / transport_path_ids"
    )

    ray_ids = np.asarray(ray_ids, dtype=int)

    n_segments = len(segments.starts)

    assert len(segments.ends) == n_segments
    assert len(segments.intensities) == n_segments

    assert len(ray_ids) == n_segments, array_mini_summary(
        "transport path IDs",
        ray_ids,
    )

    assert np.all(ray_ids >= 0), array_mini_summary(
        "transport path IDs",
        ray_ids,
    )

    unique_path_ids = np.unique(ray_ids)

    assert len(unique_path_ids) < n_segments, (
        "Fixture must contain at least one multi-segment transport path. "
        "Otherwise it does not exercise path_id != segment_index."
    )

    if segment_ids is not None:
        segment_ids = np.asarray(segment_ids, dtype=int)
        assert len(segment_ids) == n_segments, array_mini_summary(
            "segment_ids",
            segment_ids,
        )

    if path_order is not None:
        path_order = np.asarray(path_order)
        assert len(path_order) == n_segments, array_mini_summary(
            "path_order",
            path_order,
        )
