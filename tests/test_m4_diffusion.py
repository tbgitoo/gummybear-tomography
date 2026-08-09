"""Milestone 4 volumetric diffusion tests (Netgen/NGSolve path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

ngsolve = pytest.importorskip("ngsolve")
netgen = pytest.importorskip("netgen")

from gummybear.optics import (
    OpticalMaterialConfig,
    compose_hybrid_image,
    default_diffusion_coefficient,
    deposit_ray_source,
    generate_diffusion_mesh,
    make_synthetic_axis_ray,
    sample_diffuse_image,
    solve_diffusion,
)


@pytest.fixture(scope="module")
def sphere_stl(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("m4_stl")
    path = root / "sphere.stl"
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    mesh.export(path)
    return path


@pytest.fixture(scope="module")
def diff_mesh(sphere_stl, tmp_path_factory):
    cache = tmp_path_factory.mktemp("m4_cache")
    return generate_diffusion_mesh(
        sphere_stl,
        target_elements=800,
        maxh=2.5,
        cache_dir=cache,
    )


def test_generate_diffusion_mesh_netgen(diff_mesh, sphere_stl):
    surface = trimesh.load(sphere_stl, force="mesh")
    assert 50 <= diff_mesh.n_tets <= 20_000
    assert diff_mesh.n_nodes > 0
    assert diff_mesh.metadata.meshing_method == "netgen"
    assert diff_mesh.metadata.is_authoritative_geometry is False
    assert diff_mesh.netgen_mesh is not None
    sb = np.asarray(surface.bounds)
    db = diff_mesh.bounds
    # Coarse mesh bounds should be within ~20% of STL extents.
    for ax in range(3):
        extent = max(sb[1, ax] - sb[0, ax], 1e-6)
        assert abs(db[0, ax] - sb[0, ax]) < 0.25 * extent
        assert abs(db[1, ax] - sb[1, ax]) < 0.25 * extent
    assert np.all(diff_mesh.volumes > 0)


def test_deposit_source_nonnegative_and_zero_when_no_scatter(diff_mesh):
    c = diff_mesh.bounds.mean(axis=0)
    lo = diff_mesh.bounds[0]
    hi = diff_mesh.bounds[1]
    rays = make_synthetic_axis_ray(
        (c[0], c[1], lo[2] - 1.0),
        (c[0], c[1], hi[2] + 1.0),
        intensity=1.0,
    )
    mat = OpticalMaterialConfig(mu_scatter=0.5, mu_absorption=0.2)
    dep = deposit_ray_source(diff_mesh, rays, mat)
    assert np.all(dep.S_clean >= 0)
    assert dep.total_scattered >= 0
    assert abs(dep.total_scattered - float(np.sum(dep.E_scat_elem))) < 1e-10

    dep0 = deposit_ray_source(
        diff_mesh,
        rays,
        OpticalMaterialConfig(mu_scatter=0.0, mu_absorption=0.2),
    )
    assert np.allclose(dep0.S_clean, 0.0)
    assert dep0.total_scattered == 0.0


def test_solve_diffusion_responds_to_absorption(diff_mesh):
    S = np.zeros(diff_mesh.n_tets, dtype=float)
    # Localized source near mesh center.
    e0 = int(np.argmin(np.sum((diff_mesh.centroids - diff_mesh.bounds.mean(0)) ** 2, axis=1)))
    S[e0] = 1.0
    D = default_diffusion_coefficient(mu_s=0.5, mu_a=0.1)
    r_low = solve_diffusion(
        diff_mesh, S, D=D, mu_a=0.05, extrapolation_length=1.0
    )
    r_high = solve_diffusion(
        diff_mesh, S, D=D, mu_a=1.0, extrapolation_length=1.0
    )
    assert np.all(np.isfinite(r_low.Phi_nodes))
    assert np.all(np.isfinite(r_high.Phi_nodes))
    assert float(np.mean(r_high.Phi_nodes)) < float(np.mean(r_low.Phi_nodes))
    assert r_low.residual_norm is not None
    assert r_low.robin_boundary_model == "effective_refractive_boundary"


def test_diffuse_and_compose(diff_mesh):
    S = np.ones(diff_mesh.n_tets, dtype=float) * 0.1
    D = default_diffusion_coefficient(0.5, 0.1)
    solved = solve_diffusion(diff_mesh, S, D=D, mu_a=0.1, extrapolation_length=1.0)

    H = W = 16
    # Fake camera hits near the mesh center / outside.
    rng = np.random.default_rng(0)
    hits = rng.normal(loc=diff_mesh.bounds.mean(0), scale=2.0, size=(H * W, 3))
    valid = np.linalg.norm(hits - diff_mesh.bounds.mean(0), axis=1) < 4.0
    sampled = sample_diffuse_image(
        diff_mesh,
        solved.Phi_nodes,
        hits,
        valid,
        sample_shape=(H, W),
        exitance_scale=1.0,
    )
    assert sampled.I_diffuse.shape == (H, W)
    assert np.all(sampled.I_diffuse[~valid.reshape(H, W)] == 0.0)

    I_direct = np.zeros((H, W), dtype=float)
    I_direct[valid.reshape(H, W)] = 0.5
    hybrid = compose_hybrid_image(
        I_direct,
        sampled.I_diffuse,
        alpha=0.25,
        camera_mask=valid.reshape(H, W),
    )
    assert hybrid.I_total.shape == (H, W)
    assert hybrid.forward_model == "m4_refractive_diffusion"
    assert hybrid.metadata["alpha"] == 0.25
    np.testing.assert_allclose(
        hybrid.I_total[valid.reshape(H, W)],
        0.25 * I_direct[valid.reshape(H, W)] + sampled.I_diffuse[valid.reshape(H, W)],
    )
