"""Tests for Milestone 3 source rays, refraction, and face transport."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from gummybear.optics import (
    DirectionalLightConfig,
    FaceOpticalState,
    OpticalMaterialConfig,
    PointLightConfig,
    SourceSamplingParams,
    accumulate_source_coverage,
    make_source_ray_bundle,
    propagate_entry_exit_transport,
    refract_direction,
    refract_ray_bundle,
    sample_face_state_to_camera,
)
from gummybear.rays import (
    OrthographicCameraConfig,
    SourceRayBundle,
    first_visible_hits,
    first_visible_hits_with_points,
    make_camera_rays,
)


@pytest.fixture
def sphere_mesh() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    assert mesh.is_watertight
    return mesh


def test_source_ray_bundle_invariants():
    origins = np.zeros((4, 3), dtype=float)
    directions = np.tile(np.array([0.0, 0.0, -2.0]), (4, 1))
    weights = np.ones(4, dtype=float)
    bundle = SourceRayBundle(
        origins=origins,
        directions=directions,
        weights=weights,
        sample_shape=(2, 2),
    )
    assert bundle.n_rays == 4
    norms = np.linalg.norm(bundle.directions, axis=1)
    np.testing.assert_allclose(norms, 1.0)
    with pytest.raises(ValueError):
        SourceRayBundle(
            origins=origins,
            directions=directions,
            weights=np.array([-1.0, 0.0, 0.0, 0.0]),
        )


def test_refract_direction_normal_incidence_and_tir():
    normal = np.array([0.0, 0.0, 1.0])
    direction = np.array([0.0, 0.0, -1.0])
    transmitted, ok = refract_direction(direction, normal, n_from=1.0, n_to=1.33)
    assert ok
    np.testing.assert_allclose(transmitted, direction, atol=1e-10)

    # Air -> denser at grazing angle should still transmit (not NaN).
    grazing = np.array([np.sin(np.deg2rad(89.0)), 0.0, -np.cos(np.deg2rad(89.0))])
    transmitted_g, ok_g = refract_direction(grazing, normal, n_from=1.0, n_to=1.33)
    assert ok_g
    assert np.all(np.isfinite(transmitted_g))
    # Bends toward the normal: |transmitted_xy| < |incident_xy|
    assert abs(transmitted_g[0]) < abs(grazing[0])

    # Dense -> air beyond critical angle: TIR.
    # Critical angle for 1.33 -> 1.0 is arcsin(1/1.33) ≈ 48.75 deg from normal.
    theta = np.deg2rad(60.0)
    tir_dir = np.array([np.sin(theta), 0.0, -np.cos(theta)])
    _t, ok_tir = refract_direction(tir_dir, normal, n_from=1.33, n_to=1.0)
    assert not ok_tir


def test_make_source_ray_bundle_directional(sphere_mesh):
    light = DirectionalLightConfig(propagation_direction=(0.0, 0.0, -1.0), intensity=1.0)
    sampling = SourceSamplingParams(direction_n_samples=16)
    rays = make_source_ray_bundle(light=light, mesh_bbox=sphere_mesh.bounds, sampling=sampling)
    assert rays.n_rays == 16 * 16
    assert rays.sample_shape == (16, 16)
    np.testing.assert_allclose(
        rays.directions,
        np.tile(np.array([0.0, 0.0, -1.0]), (rays.n_rays, 1)),
        atol=1e-10,
    )
    assert np.all(rays.weights > 0)
    # Material must not be part of construction API (positional args only light/bbox/sampling).
    assert rays.metadata["light_type"] == "directional"


def test_first_visible_hits_accepts_source_rays(sphere_mesh):
    light = DirectionalLightConfig(propagation_direction=(0.0, 0.0, -1.0))
    rays = make_source_ray_bundle(
        light=light,
        mesh_bbox=sphere_mesh.bounds,
        sampling=SourceSamplingParams(direction_n_samples=12),
    )
    valid_mask, hit_depth, hit_faces = first_visible_hits(sphere_mesh, rays)
    assert valid_mask.shape == (rays.n_rays,)
    assert hit_depth.shape == (rays.n_rays,)
    assert hit_faces.shape == (rays.n_rays,)
    assert np.any(valid_mask)
    assert np.any(hit_faces[valid_mask] >= 0)

    # sample_shape=None still works (point / internal rays).
    point = PointLightConfig(position=(0.0, 0.0, 20.0), intensity=1.0)
    point_rays = make_source_ray_bundle(
        light=point,
        mesh_bbox=sphere_mesh.bounds,
        sampling=SourceSamplingParams(n_rays=128, seed=0),
    )
    assert point_rays.sample_shape is None
    v2, d2, f2 = first_visible_hits(sphere_mesh, point_rays)
    assert v2.shape == (point_rays.n_rays,)
    assert np.any(v2)


def test_accumulate_source_coverage_stage1(sphere_mesh):
    light = DirectionalLightConfig(propagation_direction=(0.0, 0.0, -1.0), intensity=2.0)
    rays = make_source_ray_bundle(
        light=light,
        mesh_bbox=sphere_mesh.bounds,
        sampling=SourceSamplingParams(direction_n_samples=24),
    )
    state = accumulate_source_coverage(sphere_mesh, rays)
    assert isinstance(state, FaceOpticalState)
    assert state.n_faces == len(sphere_mesh.faces)
    assert np.any(state.face_energy > 0)
    assert np.any(state.hit_count > 0)
    assert np.all(state.valid == (state.hit_count > 0))
    norms = np.linalg.norm(state.b_out[state.valid], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    # Camera sampling diagnostics remain finite.
    cam = OrthographicCameraConfig(
        camera_position=(0.0, -20.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        size=14.0,
        resolution=32,
    )
    camera_rays = make_camera_rays(cam)
    _v, _d, hit_faces = first_visible_hits(sphere_mesh, camera_rays)
    H, W = camera_rays.sample_shape
    hit_faces_img = hit_faces.reshape(H, W)
    energy_img, b_out_img, valid_img = sample_face_state_to_camera(state, hit_faces_img)
    assert energy_img.shape == (H, W)
    assert b_out_img.shape == (H, W, 3)
    assert valid_img.shape == (H, W)
    assert np.all(np.isfinite(energy_img))
    assert np.all(np.isfinite(b_out_img))


def test_first_visible_hits_with_points_helper(sphere_mesh):
    light = DirectionalLightConfig(propagation_direction=(0.0, 0.0, -1.0))
    rays = make_source_ray_bundle(
        light=light,
        mesh_bbox=sphere_mesh.bounds,
        sampling=SourceSamplingParams(direction_n_samples=8),
    )
    valid, depth, faces, points = first_visible_hits_with_points(sphere_mesh, rays)
    assert points.shape == (rays.n_rays, 3)
    assert np.all(np.isnan(points[~valid]))
    # Default first_visible_hits still returns three values.
    triple = first_visible_hits(sphere_mesh, rays)
    assert len(triple) == 3
    np.testing.assert_array_equal(triple[0], valid)


def test_refract_ray_bundle_air_to_material(sphere_mesh):
    light = DirectionalLightConfig(propagation_direction=(0.0, 0.0, -1.0), intensity=1.0)
    rays = make_source_ray_bundle(
        light=light,
        mesh_bbox=sphere_mesh.bounds,
        sampling=SourceSamplingParams(direction_n_samples=12),
    )
    material = OpticalMaterialConfig(n_refractive=1.33)
    result = refract_ray_bundle(
        sphere_mesh,
        rays,
        n_from=1.0,
        n_to=material.n_refractive,
    )

    assert result.n_input == rays.n_rays
    assert result.n_refracted == result.rays.n_rays
    assert result.n_refracted > 0
    assert result.n_refracted <= rays.n_rays
    assert result.parent_indices.shape == (result.n_refracted,)
    assert np.all(result.valid_mask[result.parent_indices])
    assert result.valid_mask.sum() == result.n_refracted

    # Origins sit just inside the mesh along the transmitted direction.
    expected_origins = (
        result.hit_points[result.parent_indices]
        + result.eps * result.rays.directions
    )
    np.testing.assert_allclose(result.rays.origins, expected_origins, atol=1e-12)
    np.testing.assert_allclose(
        result.rays.weights,
        rays.weights[result.parent_indices],
        atol=1e-12,
    )
    norms = np.linalg.norm(result.rays.directions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    # Compact internal rays should continue to hit (exit side) without self-intersection.
    exit_valid, _d, exit_faces = first_visible_hits(sphere_mesh, result.rays)
    assert np.any(exit_valid)
    assert np.any(exit_faces[exit_valid] >= 0)


def test_in_object_segments_from_refracted_rays(sphere_mesh):
    from gummybear.optics import in_object_segments_from_rays

    light = DirectionalLightConfig(propagation_direction=(0.0, 0.0, -1.0), intensity=1.0)
    rays = make_source_ray_bundle(
        light=light,
        mesh_bbox=sphere_mesh.bounds,
        sampling=SourceSamplingParams(direction_n_samples=10),
    )
    entry = refract_ray_bundle(
        sphere_mesh, rays, n_from=1.0, n_to=1.33
    )
    segments = in_object_segments_from_rays(
        sphere_mesh, entry.rays, parent_ray_ids=entry.parent_indices
    )

    assert segments.n_segments > 0
    assert segments.n_segments <= entry.rays.n_rays
    assert segments.starts.shape == (segments.n_segments, 3)
    assert segments.ends.shape == (segments.n_segments, 3)
    assert segments.intensities.shape == (segments.n_segments,)
    assert segments.ray_ids.shape == (segments.n_segments,)
    assert np.all(segments.ray_ids >= 0)
    assert segments.n_parent_rays <= segments.n_segments
    lengths = np.linalg.norm(segments.ends - segments.starts, axis=1)
    assert np.all(lengths > 0)
    # Chord through radius-5 sphere should be at most ~10.
    assert np.all(lengths < 11.0)


def test_propagate_entry_exit_transport_stage3(sphere_mesh):
    light = DirectionalLightConfig(propagation_direction=(0.0, 0.0, -1.0), intensity=1.0)
    rays = make_source_ray_bundle(
        light=light,
        mesh_bbox=sphere_mesh.bounds,
        sampling=SourceSamplingParams(direction_n_samples=20),
    )
    material = OpticalMaterialConfig(n_refractive=1.33)
    state = propagate_entry_exit_transport(
        mesh=sphere_mesh,
        source_rays=rays,
        material=material,
        apply_attenuation=False,
    )
    assert np.any(state.face_energy > 0)
    assert np.any(state.hit_count > 0)
    norms = np.linalg.norm(state.b_out[state.valid], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    assert np.all(np.isfinite(state.b_out))


def test_propagate_entry_exit_transport_ray_weights_override(sphere_mesh):
    light = DirectionalLightConfig(propagation_direction=(0.0, 0.0, -1.0), intensity=1.0)
    rays = make_source_ray_bundle(
        light=light,
        mesh_bbox=sphere_mesh.bounds,
        sampling=SourceSamplingParams(direction_n_samples=12),
    )
    material = OpticalMaterialConfig(n_refractive=1.33)
    clean = propagate_entry_exit_transport(
        sphere_mesh, rays, material, apply_attenuation=False
    )
    half_weights = rays.weights * 0.5
    half = propagate_entry_exit_transport(
        sphere_mesh, rays, material, apply_attenuation=False, ray_weights=half_weights
    )
    assert half.face_energy.sum() == pytest.approx(0.5 * clean.face_energy.sum())
