"""Tests for pinhole camera validation helpers."""

import numpy as np

from gummybear.geometry import load_stl
from gummybear.rays.camera import PinholeCameraConfig, make_pinhole_rays

from gummybear_validation.helpers.camera_helpers import (
    pinhole_camera_basis,
    pinhole_camera_framing_mesh,
    trimesh_camera_transform,
)


def test_pinhole_camera_basis_is_orthonormal():
    cam = PinholeCameraConfig(
        camera_position=(0.0, -30.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=45.0,
        resolution=64,
    )
    _eye, forward, right, up = pinhole_camera_basis(cam)

    assert np.allclose(np.linalg.norm(forward), 1.0)
    assert np.allclose(np.linalg.norm(right), 1.0)
    assert np.allclose(np.linalg.norm(up), 1.0)
    assert np.allclose(np.dot(forward, right), 0.0, atol=1e-12)
    assert np.allclose(np.dot(forward, up), 0.0, atol=1e-12)
    assert np.allclose(np.dot(right, up), 0.0, atol=1e-12)
    assert np.allclose(np.cross(right, forward), up, atol=1e-12)


def test_framing_mesh_places_camera_outside_head():
    mesh = load_stl("cad/proto_bear_head.stl")
    cam = pinhole_camera_framing_mesh(mesh, resolution=128)

    eye = np.asarray(cam.camera_position, dtype=float)
    center = np.asarray(mesh.centroid, dtype=float)
    assert eye[1] > center[1]

    forward = center - eye
    forward /= np.linalg.norm(forward)
    rays = make_pinhole_rays(cam)
    assert np.all((rays.directions @ forward) > 0.0)
    assert cam.fov_deg > 0.0


def test_trimesh_transform_matches_scene_look_at():
    mesh = load_stl("cad/proto_bear_head.stl")
    cam = pinhole_camera_framing_mesh(mesh, resolution=64)
    scene = mesh.scene()
    scene.camera.fov = (cam.fov_deg, cam.fov_deg)

    manual = trimesh_camera_transform(cam)
    eye = np.asarray(cam.camera_position, dtype=float)
    target = np.asarray(cam.look_at, dtype=float)
    rotation = np.eye(4)
    rotation[:3, :3] = manual[:3, :3]
    distance = float(np.linalg.norm(eye - target))
    auto = scene.camera.look_at(
        mesh.vertices,
        rotation=rotation,
        distance=distance,
        center=target,
    )

    assert np.allclose(manual, auto, atol=1e-12)
