"""Tests for Phi sampling localization cache (WIN 0G)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gummybear.datasets.cache_keys import (
    PHI_SAMPLING_LOCALIZATION_ALGORITHM_VERSION,
    camera_visibility_cache_key,
    phi_sampling_localization_cache_key,
    phi_sampling_localization_cache_key_payload,
)
from gummybear.datasets.source_cache import (
    PHI_SAMPLING_LOCALIZATION_PAYLOAD_SCHEMA_VERSION,
    PHI_SAMPLING_LOCALIZATION_REQUIRED_ARRAYS,
    SourceCacheStore,
)
from gummybear.optics.diffuse_sampling import (
    apply_phi_localization,
    interpolate_phi_nodes_to_points,
    localize_points_in_diffusion_mesh,
    sample_diffuse_image,
)


def _base_localization_kwargs():
    return dict(
        camera_visibility_cache_id="vis-" + ("a" * 60),
        diffusion_mesh_content_hash="mesh-hash",
        diffusion_mesh_num_nodes=100,
        diffusion_mesh_num_tets=200,
    )


def test_localization_key_stable_and_excludes_phi_optics():
    key_a = phi_sampling_localization_cache_key(**_base_localization_kwargs())
    key_b = phi_sampling_localization_cache_key(**_base_localization_kwargs())
    assert key_a == key_b
    payload = phi_sampling_localization_cache_key_payload(**_base_localization_kwargs())
    assert payload["key_kind"] == "phi_sampling_localization"
    assert payload["algorithm_version"] == PHI_SAMPLING_LOCALIZATION_ALGORITHM_VERSION
    forbidden = (
        "mu_a",
        "mu_s",
        "particle",
        "Phi",
        "extrapolation_length",
        "image_format",
        "jpeg_quality",
        "max_workers",
    )
    for name in forbidden:
        assert name not in payload
        assert name not in str(payload)


def test_localization_key_changes_with_visibility_and_mesh():
    base = _base_localization_kwargs()
    key = phi_sampling_localization_cache_key(**base)
    assert key != phi_sampling_localization_cache_key(
        **{**base, "camera_visibility_cache_id": "other"}
    )
    assert key != phi_sampling_localization_cache_key(
        **{**base, "diffusion_mesh_content_hash": "other-mesh"}
    )
    assert key != phi_sampling_localization_cache_key(
        **{**base, "barycentric_tolerance": 1e-8}
    )


def test_localization_key_links_to_visibility_id():
    vis = camera_visibility_cache_key(
        stl_sha256="stl-abc",
        camera_kind="orbit",
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
    payload = phi_sampling_localization_cache_key_payload(
        camera_visibility_cache_id=vis,
        diffusion_mesh_content_hash="mesh",
        diffusion_mesh_num_nodes=10,
        diffusion_mesh_num_tets=20,
    )
    assert payload["camera_visibility_cache_id"] == vis


def test_localization_store_roundtrip(tmp_path: Path):
    store = SourceCacheStore(tmp_path / "cache")
    cache_id = phi_sampling_localization_cache_key(**_base_localization_kwargs())
    key_payload = phi_sampling_localization_cache_key_payload(
        **_base_localization_kwargs()
    )
    mesh_identity = {
        "content_hash": "mesh-hash",
        "num_nodes": 100,
        "num_tets": 200,
    }
    arrays = {
        "sample_mode": np.array([1, 0, 2, 1], dtype=np.int8),
        "tet_id": np.array([3, -1, 1, 0], dtype=np.int64),
        "barycentric": np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        "sample_shape": np.asarray([2, 2], dtype=np.int64),
    }
    miss = store.load(
        kind="phi_sampling_localization",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=PHI_SAMPLING_LOCALIZATION_PAYLOAD_SCHEMA_VERSION,
        required_arrays=PHI_SAMPLING_LOCALIZATION_REQUIRED_ARRAYS,
        mesh_identity=mesh_identity,
    )
    assert not miss.event.hit
    store.write(
        kind="phi_sampling_localization",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=PHI_SAMPLING_LOCALIZATION_PAYLOAD_SCHEMA_VERSION,
        arrays=arrays,
        payload_metadata={"interpolation_method": "tetrahedral_barycentric"},
        mesh_identity=mesh_identity,
        workbook_provenance={"workbook_name": "test.xlsx"},
        prior_event=miss.event,
    )
    hit = store.load(
        kind="phi_sampling_localization",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=PHI_SAMPLING_LOCALIZATION_PAYLOAD_SCHEMA_VERSION,
        required_arrays=PHI_SAMPLING_LOCALIZATION_REQUIRED_ARRAYS,
        mesh_identity=mesh_identity,
    )
    assert hit.event.hit
    assert hit.arrays is not None
    np.testing.assert_array_equal(hit.arrays["sample_mode"], arrays["sample_mode"])
    np.testing.assert_array_equal(hit.arrays["tet_id"], arrays["tet_id"])
    np.testing.assert_allclose(hit.arrays["barycentric"], arrays["barycentric"])


def test_apply_localization_matches_interpolate():
    """Behavioral parity: localize once + apply == interpolate_phi_nodes_to_points."""
    from gummybear.optics.diffusion_mesh import DiffusionMesh, DiffusionMeshMetadata

    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    centroids = nodes[tets].mean(axis=1)
    # Simple tet volumes (positive placeholders)
    volumes = np.array([1.0 / 6.0, 1.0 / 6.0], dtype=float)
    metadata = DiffusionMeshMetadata(
        stl_hash="test",
        geometry_id="unit",
        num_elements=2,
        num_nodes=5,
    )
    mesh = DiffusionMesh(
        nodes=nodes,
        tets=tets,
        centroids=centroids,
        volumes=volumes,
        metadata=metadata,
        netgen_mesh=None,
    )

    Phi = np.linspace(1.0, 2.0, mesh.n_nodes)
    points = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [np.nan, np.nan, np.nan],
        ],
        dtype=float,
    )
    valid = np.array([True, True, False])
    direct = interpolate_phi_nodes_to_points(mesh, Phi, points, valid_mask=valid)
    localization = localize_points_in_diffusion_mesh(mesh, points, valid_mask=valid)
    applied = apply_phi_localization(mesh, Phi, localization)
    np.testing.assert_allclose(direct, applied, equal_nan=True)

    shape = (1, 3)
    img_a = sample_diffuse_image(
        mesh, Phi, points, valid, shape, exitance_scale=2.0, interpolate=True
    ).I_diffuse
    img_b = sample_diffuse_image(
        mesh,
        Phi,
        points,
        valid,
        shape,
        exitance_scale=2.0,
        interpolate=True,
        localization=localization,
    ).I_diffuse
    np.testing.assert_allclose(img_a, img_b)


def test_load_or_compute_localization_skips_search_on_hit(tmp_path: Path, monkeypatch):
    from gummybear.datasets.sequence_generation import _load_or_compute_phi_localization
    from gummybear.optics.diffusion_mesh import DiffusionMesh, DiffusionMeshMetadata

    calls = {"n": 0}
    real_localize = localize_points_in_diffusion_mesh

    def counting_localize(*args, **kwargs):
        calls["n"] += 1
        return real_localize(*args, **kwargs)

    monkeypatch.setattr(
        "gummybear.datasets.sequence_generation.localize_points_in_diffusion_mesh",
        counting_localize,
    )

    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    mesh = DiffusionMesh(
        nodes=nodes,
        tets=tets,
        centroids=nodes[tets].mean(axis=1),
        volumes=np.array([1.0 / 6.0]),
        metadata=DiffusionMeshMetadata(
            stl_hash="t",
            geometry_id="g",
            num_elements=1,
            num_nodes=4,
        ),
    )
    points = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.2]])
    valid = np.ones(4, dtype=bool)
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
    store = SourceCacheStore(tmp_path / "cache")
    vis_id = "vis-" + ("b" * 60)

    first = _load_or_compute_phi_localization(
        diff_mesh=mesh,
        hit_points=points,
        valid_mask=valid,
        sample_shape=(2, 2),
        camera_visibility_cache_id=vis_id,
        job=job,
        cache_store=store,
        force_recompute=False,
    )
    assert first[1].status == "miss"
    assert calls["n"] == 1

    second = _load_or_compute_phi_localization(
        diff_mesh=mesh,
        hit_points=points,
        valid_mask=valid,
        sample_shape=(2, 2),
        camera_visibility_cache_id=vis_id,
        job=job,
        cache_store=store,
        force_recompute=False,
    )
    assert second[1].status == "hit"
    assert calls["n"] == 1
    np.testing.assert_array_equal(first[0].tet_id, second[0].tet_id)
