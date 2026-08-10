"""Installed pytest checks for generated sequence artifact contracts.

Notebook / protocol: M6.3
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gummybear.datasets.source_cache import CacheEvent, SourceCacheStore
from gummybear_validation.milestone_06.validate_sequence import (
    validate_generated_sequence,
)


def _write_validation_fixture(root: Path) -> Path:
    """Write one tiny valid sequence without invoking generation physics."""
    sequence_id = "m6_installed_validation"
    sequence_dir = root / sequence_id
    for role in ("clean", "particle", "observed"):
        (sequence_dir / role).mkdir(parents=True, exist_ok=True)

    stem = f"{sequence_id}_frame_0000_angle_+0000.00"
    clean = np.full((8, 8), 20, dtype=np.uint8)
    particle = clean.copy()
    particle[2:6, 2:6] = 80
    filenames = {}
    for role, pixels in (
        ("clean", clean),
        ("particle", particle),
        ("observed", particle),
    ):
        relative_name = f"{role}/{stem}.jpg"
        Image.fromarray(pixels, mode="L").save(
            sequence_dir / relative_name,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        filenames[role] = relative_name
        raw_relative = f"{role}/{stem}.raw.tif"
        Image.fromarray(pixels.astype(np.float32), mode="F").save(
            sequence_dir / raw_relative,
            format="TIFF",
        )
        filenames[f"{role}_raw"] = raw_relative

    def setup(sheet: str) -> dict:
        return {
            "source_sheet": sheet,
            "source_excel_row": 2,
            "workbook_name": "m6_test.xlsx",
            "workbook_sheet": sheet,
        }

    manifest = {
        "schema_version": "1.2-m6-draft",
        "generator_version": "m6-draft",
        "sequence_id": sequence_id,
        "created_utc": "2026-07-19T09:00:00+00:00",
        "split": "train",
        "seed": 42,
        "forward_model_tier": "m5_refractive_diffusion_particle_perturbation",
        "representation": {
            "image_format": "jpg",
            "jpeg_quality": 95,
            "image_domain": "camera_intensity",
            "composition_domain": "linear_camera_intensity_before_jpeg",
            "composition_policy": "pre_jpeg_numeric_arrays",
            "anomaly_definition": "particle_minus_clean",
            "observed_definition": "particle_no_corruption",
            "pixel_orientation": {
                "camera_up": "image_top",
                "transform_from_camera_sample_grid": "flip_axis_0",
            },
            "anomaly_preview": {
                "authoritative": False,
                "format": None,
                "mapping": None,
            },
        },
        "roles": {
            "clean": "clean",
            "particle": "particle",
            "observed": "observed",
        },
        "phantom": {
            "phantom_id": "proto_bear",
            "stl_path": "cad/proto_bear.stl",
            "stl_sha256": "a" * 64,
        },
        "workbook": {
            "workbook_path": "configs/m6/m6_test.xlsx",
            "sha256": "b" * 64,
            "sequence_sheet": "sequences",
            "sequence_excel_row": 2,
        },
        "setups": {
            "optical": setup("optical_setups"),
            "particle": setup("particles"),
            "diffusion": {
                **setup("diffusion_setups"),
                "effective": {"D": 0.8, "mu_a": 0.1},
            },
            "camera": setup("camera_schedules"),
            "corruption": setup("corruptions"),
        },
        "caches": {
            "clean_optical_cache_id": "c" * 64,
            "particle_source_cache_id": "d" * 64,
            "persistent_cache_used": False,
            "diffusion_operator_cache": None,
        },
        "generation": {
            "package_version": "0.0.1.dev0",
            "max_workers": 1,
            "runtime_settings": {},
            "stage_seconds": {"write": 0.01},
            "diagnostics": {
                "n_source_rays": 16,
                "source_assignment": "attenuated_chord",
            },
        },
        "frames": [
            {
                "frame_index": 0,
                "angle_deg": 0.0,
                "axis": [0.0, 0.0, 1.0],
                "camera_kind": "orbit",
                "camera_position": [0.0, -80.0, 2.5],
                "look_at": [0.0, 0.0, 2.5],
                "up": [0.0, 0.0, 1.0],
                "resolution": [8, 8],
                "fov_deg": 35.0,
                "filenames": filenames,
            }
        ],
        "validation": {
            "role_alignment": "checked_before_write",
            "anomaly_identity": "particle_minus_clean_pre_jpeg",
            "post_jpeg_identity_checked": False,
        },
    }
    (sequence_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sequence_dir


@pytest.mark.milestone("Milestone 6.3")
@pytest.mark.proves(
    "A coherent historical M6.2 sequence (persistent_cache_used=False) passes "
    "manifest, filename, role-alignment, portability, provenance, and "
    "decoded-image validation."
)
def test_generated_sequence_artifact_contract(tmp_path):
    """Check a minimal valid sequence passes artifact validation.

    **Pass:** ``validate_generated_sequence`` returns ``ok`` with expected
    frame/role counts and ``persistent_cache_used=False`` in the fixture manifest.

    Notebook / protocol: M6.3
    """
    sequence_dir = _write_validation_fixture(tmp_path)

    result = validate_generated_sequence(sequence_dir)

    assert result.ok, result.errors
    assert result.frame_count == 1
    assert result.role_names == ("clean", "particle", "observed")
    manifest = json.loads(
        (sequence_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["caches"]["persistent_cache_used"] is False


@pytest.mark.milestone("Milestone 6.3")
@pytest.mark.proves(
    "Machine-local workbook paths are rejected from portable M6 manifests."
)
def test_generated_sequence_rejects_absolute_workbook_path(tmp_path):
    """Reject manifests whose workbook path is machine-local absolute.

    **Pass:** validation fails with an error mentioning the workbook path.

    Notebook / protocol: M6.3
    """
    sequence_dir = _write_validation_fixture(tmp_path)
    manifest_path = sequence_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["workbook"]["workbook_path"] = "/Users/example/private/plan.xlsx"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = validate_generated_sequence(sequence_dir)

    assert not result.ok
    assert any("workbook path" in error for error in result.errors)


@pytest.mark.milestone("Milestone 6.3")
@pytest.mark.proves(
    "Decoded JPG identity is explicitly approximate and never presented as an "
    "exact pre-JPEG composition check."
)
def test_generated_sequence_labels_composition_check_domain(tmp_path):
    """Document that display JPEG preview identity checks are approximate, not pre-JPEG exact.

    **Pass:** validation succeeds with ``post_jpeg_identity_checked`` True,
    ``pre_jpeg_identity_checked`` False, and a warning that display JPEG preview checks are not
    exact pre-compression residuals.

    Notebook / protocol: M6.3
    """
    sequence_dir = _write_validation_fixture(tmp_path)

    result = validate_generated_sequence(sequence_dir)

    assert result.ok
    assert result.post_jpeg_identity_checked is True
    assert result.pre_jpeg_identity_checked is False
    assert any("not an exact pre-JPEG" in warning for warning in result.warnings)


@pytest.mark.milestone("Milestone 6.4")
@pytest.mark.proves(
    "A completed numeric source-cache pair is reusable only with matching key, "
    "schema, required arrays, and diffusion-mesh alignment."
)
def test_source_cache_roundtrip_and_mesh_alignment(tmp_path):
    """Verify source-cache hit/miss semantics and diffusion-mesh alignment.

    **Pass:** identical key/mesh loads as ``hit``; mismatched mesh content hash
    returns ``miss_mesh_alignment_mismatch``.

    Notebook / protocol: M6.4
    """
    store = SourceCacheStore(tmp_path)
    cache_id = "e" * 64
    key_payload = {"algorithm_version": "installed-cache-test-v1"}
    mesh_identity = {
        "content_hash": "mesh-content-a",
        "num_nodes": 4,
        "num_tets": 1,
    }
    event = CacheEvent(
        kind="clean_optical",
        cache_id=cache_id,
        status="miss",
        reason="miss_not_found",
        payload_path=None,
        sidecar_path=None,
    )
    store.write(
        kind="clean_optical",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version="installed-cache-schema-v1",
        arrays={"values": np.asarray([1.0, 2.0])},
        payload_metadata={"description": "installed validation fixture"},
        mesh_identity=mesh_identity,
        workbook_provenance={},
        prior_event=event,
    )

    hit = store.load(
        kind="clean_optical",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version="installed-cache-schema-v1",
        required_arrays=("values",),
        mesh_identity=mesh_identity,
    )
    mismatch = store.load(
        kind="clean_optical",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version="installed-cache-schema-v1",
        required_arrays=("values",),
        mesh_identity={**mesh_identity, "content_hash": "mesh-content-b"},
    )

    assert hit.event.status == "hit"
    assert hit.arrays is not None
    assert np.array_equal(hit.arrays["values"], np.asarray([1.0, 2.0]))
    assert mismatch.event.status == "miss"
    assert mismatch.event.reason == "miss_mesh_alignment_mismatch"
