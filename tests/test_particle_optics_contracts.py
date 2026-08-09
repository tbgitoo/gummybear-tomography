"""Particle / transport optics contracts (no FEM).

Covers geometry hits, ParticleSet overlap, clean/dirty energy roles,
Beer–Lambert chord deposition, and source-correction identity.
"""

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
    ParticleOverlapError,
    ParticleSet,
    ParticleSphere,
    build_affected_transport_pairs,
    compute_transport_source_correction,
    deposit_particle_scatter_sources,
    partition_particle_loss,
    segment_sphere_intersection,
)


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


def _two_tet_mesh() -> DiffusionMesh:
    """Two tets sharing face (0,1,2); chord along +z crosses both."""
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
            abs(
                np.linalg.det(
                    np.column_stack(
                        (nodes[1] - nodes[0], nodes[2] - nodes[0], nodes[3] - nodes[0])
                    )
                )
            )
            / 6.0,
            abs(
                np.linalg.det(
                    np.column_stack(
                        (nodes[1] - nodes[0], nodes[2] - nodes[0], nodes[4] - nodes[0])
                    )
                )
            )
            / 6.0,
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


def test_partition_particle_loss_uses_relative_mu():
    E_abs, E_scat = partition_particle_loss(10.0, mu_abs=2.0, mu_scat=8.0)
    assert E_abs == pytest.approx(2.0)
    assert E_scat == pytest.approx(8.0)


def test_geometrically_overlapping_particles_rejected_by_default():
    with pytest.raises(ParticleOverlapError, match="overlapping spheres"):
        ParticleSet.from_particles(
            [
                ParticleSphere(center=(-0.2, 0.0, 0.0), radius=0.5),
                ParticleSphere(center=(0.2, 0.0, 0.0), radius=0.5),
            ]
        )


def test_single_pure_absorber_lowers_dirty_intensity_without_scatter_source():
    segments = make_synthetic_axis_ray((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    particles = [ParticleSphere(center=(0.0, 0.0, 0.0), radius=0.5, mu_abs=2.0)]
    pair = build_affected_transport_pairs(segments, particles).pairs[0]
    assert pair.dirty_output_intensity < pair.clean_output_intensity
    assert pair.total_E_abs > 0.0
    assert pair.total_E_scat == pytest.approx(0.0)


def test_single_pure_scatterer_generates_positive_scatter_source():
    segments = make_synthetic_axis_ray((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    particles = [ParticleSphere(center=(0.0, 0.0, 0.0), radius=0.5, mu_scat=2.0)]
    pair = build_affected_transport_pairs(segments, particles).pairs[0]
    assert pair.dirty_output_intensity < pair.clean_output_intensity
    assert pair.total_E_abs == pytest.approx(0.0)
    assert pair.total_E_scat > 0.0


def test_mixed_particle_energy_partition_closes():
    segments = make_synthetic_axis_ray((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    particles = [
        ParticleSphere(center=(0.0, 0.0, 0.0), radius=0.5, mu_abs=1.0, mu_scat=3.0)
    ]
    event = build_affected_transport_pairs(segments, particles).pairs[0].particle_scatter_source_events[0]
    assert event.E_abs + event.E_scat == pytest.approx(event.I_before - event.I_after)
    assert event.E_scat / event.E_abs == pytest.approx(3.0)


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
    I_before = 1.0
    length = float(np.linalg.norm(exit_ - entry))
    I_after = I_before * np.exp(-mu_scat * length)
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
        metadata={"mu_abs": 0.0, "mu_scat": mu_scat, "path_length": length},
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
    half = 0.5
    expected_entry = I_before * (1.0 - np.exp(-mu_scat * half))
    expected_exit = I_before * np.exp(-mu_scat * half) * (1.0 - np.exp(-mu_scat * half))
    assert deposition.delta_E_particle_scat_elem[1] == pytest.approx(expected_entry)
    assert deposition.delta_E_particle_scat_elem[0] == pytest.approx(expected_exit)
    assert deposition.delta_E_particle_scat_elem[1] > deposition.delta_E_particle_scat_elem[0]


def test_source_correction_identity_and_inspectable_split():
    mesh = _single_tet_mesh()
    c = mesh.centroids[0]
    segments = RaySegmentBundle(
        starts=np.array([[-0.5, c[1], c[2]]]),
        ends=np.array([[1.5, c[1], c[2]]]),
        intensities=np.array([1.0]),
        ray_ids=np.array([11]),
    )
    particles = [ParticleSphere(center=c, radius=0.1, mu_abs=1.0, mu_scat=2.0)]
    pairs = build_affected_transport_pairs(segments, particles, mu_s=1.0, mu_a=0.0)
    clean = deposit_ray_source(mesh, segments, mu_s=1.0, mu_a=0.0)
    source = compute_transport_source_correction(
        mesh, pairs, clean.E_scat_elem, mu_s=1.0, mu_a=0.0
    )
    assert np.allclose(
        source.delta_E_transport_elem,
        source.delta_E_background_elem + source.delta_E_particle_scat_elem,
    )
    assert np.allclose(
        source.E_particle_elem,
        source.E_clean_elem + source.delta_E_transport_elem,
    )
    assert source.metadata["source_model"] == "affected_transport_pair_delta"
