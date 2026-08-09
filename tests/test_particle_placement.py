"""Tests for offline random particle centre placement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from gummybear.particles import ParticleSet, ParticleSphere, sample_random_centers_in_mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_BEAR = REPO_ROOT / "cad" / "proto_bear.stl"


def _unit_box() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[2.0, 2.0, 2.0])


def test_sample_count_shape_and_inside_box():
    mesh = _unit_box()
    centers = sample_random_centers_in_mesh(mesh, 20, seed=7)
    assert centers.shape == (20, 3)
    assert np.all(mesh.contains(centers))


def test_sample_accepts_path_str_and_pathlib():
    centers_str = sample_random_centers_in_mesh(str(PROTO_BEAR), 5, seed=11)
    centers_path = sample_random_centers_in_mesh(PROTO_BEAR, 5, seed=11)
    np.testing.assert_allclose(centers_str, centers_path)
    mesh = trimesh.load(PROTO_BEAR, force="mesh")
    assert np.all(mesh.contains(centers_path))


def test_sample_seed_reproducible():
    mesh = _unit_box()
    a = sample_random_centers_in_mesh(mesh, 8, seed=42)
    b = sample_random_centers_in_mesh(mesh, 8, seed=42)
    c = sample_random_centers_in_mesh(mesh, 8, seed=43)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_sample_zero_returns_empty():
    centers = sample_random_centers_in_mesh(_unit_box(), 0, seed=0)
    assert centers.shape == (0, 3)


def test_sample_min_separation_and_radius_default():
    mesh = _unit_box()
    centers = sample_random_centers_in_mesh(
        mesh,
        4,
        seed=3,
        radius=0.25,
    )
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            assert np.linalg.norm(centers[i] - centers[j]) >= 0.5 - 1e-12

    particles = ParticleSet.from_particles(
        [
            ParticleSphere(center=c, radius=0.25, particle_id=f"p{i}")
            for i, c in enumerate(centers)
        ]
    )
    assert particles.validate()


def test_sample_raises_when_impossible():
    mesh = _unit_box()
    with pytest.raises(ValueError, match="Failed to sample"):
        sample_random_centers_in_mesh(
            mesh,
            5,
            seed=1,
            min_center_separation=10.0,
            max_attempts=200,
        )


def test_non_watertight_rejected():
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    with pytest.raises(ValueError, match="watertight"):
        sample_random_centers_in_mesh(mesh, 1, seed=0)
