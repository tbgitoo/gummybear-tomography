"""Empty transport-intersection particles must warn, not abort generation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gummybear.datasets.generation_plan import ParticleSetupConfig
from gummybear.datasets.sequence_generation import (
    DefaultSmokePhysicsBackend,
    SmokeRuntimeSettings,
    _CleanState,
)
from gummybear.optics.diffusion_mesh import DiffusionMesh, DiffusionMeshMetadata
from gummybear.optics.material import OpticalMaterialConfig
from gummybear.optics.source_deposition import RaySegmentBundle
from gummybear.particles.perturbation import AffectedTransportPairResult


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


def test_prepare_particle_empty_transport_intersection_warns(monkeypatch):
    empty = AffectedTransportPairResult(
        pairs=(),
        affected_path_ids=(),
        affected_segment_indices=np.zeros(0, dtype=int),
        total_E_abs=0.0,
        total_E_scat=0.0,
    )
    monkeypatch.setattr(
        "gummybear.datasets.sequence_generation.build_affected_transport_pairs",
        lambda *args, **kwargs: empty,
    )

    mesh = _single_tet_mesh()
    e_clean = np.array([0.5], dtype=float)
    clean = _CleanState(
        surface_mesh=None,
        diff_mesh=mesh,
        material=OpticalMaterialConfig(
            n_refractive=1.4,
            mu_absorption=0.01,
            mu_scatter=0.2,
        ),
        light=None,
        source_rays=SimpleNamespace(weights=np.ones(2, dtype=float)),
        refracted=None,
        segments=RaySegmentBundle(
            starts=np.zeros((0, 3), dtype=float),
            ends=np.zeros((0, 3), dtype=float),
            intensities=np.zeros(0, dtype=float),
            ray_ids=np.zeros(0, dtype=int),
        ),
        deposition=SimpleNamespace(E_scat_elem=e_clean, S_clean=e_clean * 6.0),
    )
    particle = ParticleSetupConfig(
        particle_setup_id="shadow_p0",
        particle_kind="sphere",
        center_x=0.25,
        center_y=0.25,
        center_z=0.25,
        radius=0.05,
        mu_s_particle=10.0,
        mu_a_particle=0.0,
        refractive_index_particle=1.4,
        placement_mode="fixed",
        seed=None,
    )
    job = SimpleNamespace(
        sequence_id="shadow_seq",
        particles=(particle,),
        particle=particle,
        particle_group_id="",
    )

    backend = DefaultSmokePhysicsBackend()
    with pytest.warns(UserWarning, match="intersects no source transport paths"):
        state = backend.prepare_particle(job, clean, SmokeRuntimeSettings())

    assert state.no_transport_intersection is True
    assert state.notes
    np.testing.assert_allclose(state.source_correction.S_particle, e_clean * 6.0)
    assert len(state.pair_result.affected_path_ids) == 0

    arrays, metadata = backend.serialize_particle(state)
    assert metadata["no_transport_intersection"] is True
    assert metadata["n_affected_paths"] == 0
    restored = backend.restore_particle(job, arrays, metadata)
    assert restored.no_transport_intersection is True
