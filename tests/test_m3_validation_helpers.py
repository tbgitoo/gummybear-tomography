"""Tests for M3 validation notebook helpers."""

import numpy as np
import trimesh

from gummybear.optics import DirectionalLightConfig, SourceSamplingParams, make_source_ray_bundle
from gummybear.optics.face_transport import FaceOpticalState
from gummybear.rays import first_visible_hits

from gummybear_validation.helpers.m3_transport import (
    assert_source_hit_contract,
    assert_source_ray_bundle_invariants,
)
from gummybear_validation.plotting.m3_face_transport import sample_face_state_camera_images


def test_assert_source_ray_bundle_invariants_unit_sphere():
    mesh = trimesh.primitives.Sphere(radius=1.0)
    light = DirectionalLightConfig(propagation_direction=(0.0, 0.0, -1.0), intensity=1.0)
    bundle = make_source_ray_bundle(
        light,
        mesh_bbox=mesh.bounds,
        sampling=SourceSamplingParams(direction_n_samples=8),
    )
    assert_source_ray_bundle_invariants(bundle)


def test_sample_face_state_camera_images_shape():
    mesh = trimesh.primitives.Box()
    n_faces = len(mesh.faces)
    state = FaceOpticalState(
        face_energy=np.linspace(0.0, 1.0, n_faces),
        b_out=np.tile(np.array([0.0, 0.0, 1.0]), (n_faces, 1)),
        hit_count=np.ones(n_faces, dtype=np.int64),
        valid=np.ones(n_faces, dtype=bool),
    )
    from gummybear.rays.camera import PinholeCameraConfig, make_pinhole_rays

    cam = PinholeCameraConfig(
        camera_position=(0.0, -5.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        fov_deg=45.0,
        resolution=4,
    )
    camera_rays = make_pinhole_rays(cam)
    _valid, _depth, hit_faces = first_visible_hits(mesh, camera_rays)
    images = sample_face_state_camera_images(state, hit_faces, camera_rays.sample_shape)
    assert images["energy"].shape == (4, 4)
    assert images["b_out_z"].shape == (4, 4)


def test_assert_source_hit_contract_on_box():
    mesh = trimesh.primitives.Box()
    light = DirectionalLightConfig(propagation_direction=(0.0, 0.0, -1.0), intensity=1.0)
    bundle = make_source_ray_bundle(
        light,
        mesh_bbox=mesh.bounds,
        sampling=SourceSamplingParams(direction_n_samples=16),
    )
    valid, depth, faces = first_visible_hits(mesh, bundle)
    assert_source_hit_contract(bundle, valid, depth, faces)
