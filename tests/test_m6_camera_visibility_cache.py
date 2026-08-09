"""Tests for camera×mesh visibility / hit-geometry cache (WIN 0F)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gummybear.datasets.cache_keys import (
    CAMERA_VISIBILITY_ALGORITHM_VERSION,
    camera_visibility_cache_key,
    camera_visibility_cache_key_payload,
    clean_optical_cache_key,
)
from gummybear.datasets.source_cache import (
    CAMERA_VISIBILITY_ALLOW_NONFINITE_ARRAYS,
    CAMERA_VISIBILITY_PAYLOAD_SCHEMA_VERSION,
    CAMERA_VISIBILITY_REQUIRED_ARRAYS,
    SourceCacheStore,
)


def _base_visibility_kwargs():
    return dict(
        stl_sha256="stl-abc",
        camera_kind="pinhole",
        fov_deg=35.0,
        resolution_x=128,
        resolution_y=128,
        frame_index=0,
        angle_deg=0.0,
        axis_x=0.0,
        axis_y=0.0,
        axis_z=1.0,
        distance=80.0,
        elevation_deg=0.0,
    )


def test_visibility_key_stable_and_excludes_optics_and_particles():
    key_a = camera_visibility_cache_key(**_base_visibility_kwargs())
    key_b = camera_visibility_cache_key(**_base_visibility_kwargs())
    assert key_a == key_b
    assert len(key_a) == 64

    payload = camera_visibility_cache_key_payload(**_base_visibility_kwargs())
    assert payload["key_kind"] == "camera_visibility"
    assert payload["algorithm_version"] == CAMERA_VISIBILITY_ALGORITHM_VERSION
    assert payload["look_at_policy"] == "mesh_bounds_centroid"
    forbidden = (
        "mu_a",
        "mu_s",
        "particle",
        "particles",
        "Phi",
        "extrapolation_length",
        "robin_boundary_model",
        "image_format",
        "jpeg_quality",
        "max_workers",
        "source_intensity",
        "illumination",
    )
    blob = str(payload)
    for name in forbidden:
        assert name not in payload
        assert f"'{name}'" not in blob


def test_visibility_key_changes_with_pose_resolution_fov_and_stl():
    base = _base_visibility_kwargs()
    key = camera_visibility_cache_key(**base)
    assert key != camera_visibility_cache_key(**{**base, "angle_deg": 10.0})
    assert key != camera_visibility_cache_key(**{**base, "resolution_x": 64, "resolution_y": 64})
    assert key != camera_visibility_cache_key(**{**base, "fov_deg": 40.0})
    assert key != camera_visibility_cache_key(**{**base, "stl_sha256": "other"})
    assert key != camera_visibility_cache_key(**{**base, "distance": 90.0})


def test_visibility_key_independent_of_clean_optical_key_inputs():
    visibility = camera_visibility_cache_key(**_base_visibility_kwargs())
    clean = clean_optical_cache_key(
        stl_sha256="stl-abc",
        illumination_kind="point",
        light_position_x=1.0,
        light_position_y=2.0,
        light_position_z=3.0,
        num_source_rays=512,
        mu_s=0.3,
        mu_a=0.1,
        refractive_index=1.33,
        source_deposition_method="exact_ray_tet_intervals",
    )
    assert visibility != clean


def test_visibility_store_roundtrip_hit_on_second_load(tmp_path: Path):
    store = SourceCacheStore(tmp_path / "cache")
    cache_id = camera_visibility_cache_key(**_base_visibility_kwargs())
    key_payload = camera_visibility_cache_key_payload(**_base_visibility_kwargs())
    mesh_identity = {
        "identity_kind": "camera_visibility_surface",
        "look_at_policy": "mesh_bounds_centroid",
        "stl_sha256": "stl-abc",
    }
    arrays = {
        "valid_mask": np.array([True, False, True, False]),
        "hit_depth": np.array([1.0, np.nan, 2.0, np.nan], dtype=float),
        "hit_faces": np.array([0, -1, 3, -1], dtype=np.int64),
        "hit_points": np.array(
            [
                [0.0, 0.0, 0.0],
                [np.nan, np.nan, np.nan],
                [1.0, 2.0, 3.0],
                [np.nan, np.nan, np.nan],
            ],
            dtype=float,
        ),
        "sample_shape": np.asarray([2, 2], dtype=np.int64),
    }
    miss = store.load(
        kind="camera_visibility",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=CAMERA_VISIBILITY_PAYLOAD_SCHEMA_VERSION,
        required_arrays=CAMERA_VISIBILITY_REQUIRED_ARRAYS,
        mesh_identity=mesh_identity,
        allow_nonfinite_arrays=CAMERA_VISIBILITY_ALLOW_NONFINITE_ARRAYS,
    )
    assert not miss.event.hit
    written = store.write(
        kind="camera_visibility",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=CAMERA_VISIBILITY_PAYLOAD_SCHEMA_VERSION,
        arrays=arrays,
        payload_metadata={"look_at_policy": "mesh_bounds_centroid"},
        mesh_identity=mesh_identity,
        workbook_provenance={"workbook_name": "test.xlsx"},
        prior_event=miss.event,
    )
    assert written.payload_path is not None
    assert Path(written.payload_path).is_file()
    assert (tmp_path / "cache" / "camera_visibility").is_dir()

    hit = store.load(
        kind="camera_visibility",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=CAMERA_VISIBILITY_PAYLOAD_SCHEMA_VERSION,
        required_arrays=CAMERA_VISIBILITY_REQUIRED_ARRAYS,
        mesh_identity=mesh_identity,
        allow_nonfinite_arrays=CAMERA_VISIBILITY_ALLOW_NONFINITE_ARRAYS,
    )
    assert hit.event.hit
    assert hit.arrays is not None
    np.testing.assert_array_equal(hit.arrays["valid_mask"], arrays["valid_mask"])
    np.testing.assert_allclose(
        hit.arrays["hit_depth"],
        arrays["hit_depth"],
        equal_nan=True,
    )
    np.testing.assert_array_equal(hit.arrays["hit_faces"], arrays["hit_faces"])


def test_load_or_compute_visibility_skips_raycast_on_hit(tmp_path: Path, monkeypatch):
    from gummybear.datasets.generation_plan import CameraPose
    from gummybear.datasets.sequence_generation import (
        SmokeRuntimeSettings,
        _load_or_compute_camera_visibility,
    )

    calls = {"n": 0}

    def fake_hits(mesh, rays):
        calls["n"] += 1
        n = int(np.prod(rays.sample_shape))
        return (
            np.ones(n, dtype=bool),
            np.ones(n, dtype=float),
            np.zeros(n, dtype=np.int64),
            np.zeros((n, 3), dtype=float),
        )

    monkeypatch.setattr(
        "gummybear.datasets.sequence_generation.first_visible_hits_with_points",
        fake_hits,
    )

    class _Rays:
        sample_shape = (2, 2)

    job = type(
        "Job",
        (),
        {
            "stl_sha256": "stl-abc",
            "stl_path": "cad/proto_bear.stl",
            "workbook_path": "configs/m6/m6_generation_plan.xlsx",
            "workbook_sha256": "wb",
            "source_excel_row": 2,
        },
    )()
    pose = CameraPose(
        frame_index=0,
        angle_deg=0.0,
        axis=(0.0, 0.0, 1.0),
        distance=80.0,
        elevation_deg=0.0,
        resolution_x=2,
        resolution_y=2,
        camera_kind="pinhole",
    )
    settings = SmokeRuntimeSettings()
    store = SourceCacheStore(tmp_path / "cache")

    first = _load_or_compute_camera_visibility(
        surface_mesh=object(),
        rays=_Rays(),
        job=job,
        pose=pose,
        settings=settings,
        cache_store=store,
        force_recompute=False,
    )
    assert first[4].status == "miss"
    assert calls["n"] == 1

    second = _load_or_compute_camera_visibility(
        surface_mesh=object(),
        rays=_Rays(),
        job=job,
        pose=pose,
        settings=settings,
        cache_store=store,
        force_recompute=False,
    )
    assert second[4].status == "hit"
    assert calls["n"] == 1
    np.testing.assert_array_equal(first[0], second[0])
