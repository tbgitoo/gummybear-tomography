"""Tests for M2B validation helpers."""

import numpy as np
import trimesh

from gummybear.geometry import load_stl
from gummybear.optics import PointLightConfig
from gummybear.rays.camera import PinholeCameraConfig, make_pinhole_rays

from gummybear_validation.helpers import compute_m2b_debug_proxy
from gummybear_validation.plotting import (
    collect_m2b_proxy_diagnostics,
    face_centroid_channels,
    sample_face_scalars_to_images,
)


def test_face_centroid_channels_match_mesh():
    mesh = trimesh.primitives.Box()
    x, y, z, idx = face_centroid_channels(mesh)
    assert x.shape == (len(mesh.faces),)
    assert np.allclose(idx, np.arange(len(mesh.faces)))


def test_m2b_debug_proxy_factorization_on_box():
    mesh = trimesh.primitives.Box()
    camera = PinholeCameraConfig(
        camera_position=(0.0, -5.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=45.0,
        resolution=32,
    )
    rays = make_pinhole_rays(camera)
    light = PointLightConfig(position=(0.0, 0.0, 5.0), intensity=1.0, falloff="none")

    result = compute_m2b_debug_proxy(mesh, rays, light, mu=0.1)
    diag = collect_m2b_proxy_diagnostics(
        result.field,
        result.L_proxy,
        result.T_face,
        result.face_source_intensity,
        result.I_proxy,
    )

    valid = result.field.valid_mask
    recomposed = diag["source_sampled"] * diag["T_sampled"] * diag["g_bv"]
    assert np.allclose(recomposed[valid], result.I_proxy[valid], rtol=1e-5, equal_nan=True)


def test_sample_face_scalars_to_images_shape():
    mesh = load_stl("cad/proto_bear.stl")
    camera = PinholeCameraConfig(
        camera_position=(0.0, -40.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        fov_deg=35.0,
        resolution=16,
    )
    rays = make_pinhole_rays(camera)
    from gummybear.rays.visibility import first_visible_hits

    _valid, _depth, hit_faces = first_visible_hits(mesh, rays)
    channels = face_centroid_channels(mesh)
    images = sample_face_scalars_to_images(hit_faces, channels[:2], rays.sample_shape)
    assert images[0].shape == rays.sample_shape
