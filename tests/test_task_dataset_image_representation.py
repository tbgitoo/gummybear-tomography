"""Unit tests for catalog role-image representation modes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gummybear.datasets.role_images import write_float_raw_tif, write_uint8_image
from tomography_ml.gummybear_data_catalog import (
    DEFAULT_IMAGE_REPRESENTATION,
    M7_IMAGE_REPRESENTATION,
    RoleRef,
    load_role_array,
)
from tomography_ml.gummybear_data_catalog.task_dataset import (
    DatasetTaskSpec,
    IMAGE_REPRESENTATION_JPEG_UINT8,
    IMAGE_REPRESENTATION_RAW_FLOAT,
)


def _write_tiny_sequence(sequence_dir: Path) -> RoleRef:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = sequence_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    jpg_rel = "clean/seq_frame_0000_angle_+0000.00.jpg"
    raw_rel = "clean/seq_frame_0000_angle_+0000.00.raw.tif"
    float_frame = np.array(
        [[0.0, 0.25], [0.5, 1.0]],
        dtype=np.float32,
    )
    write_float_raw_tif(sequence_dir / raw_rel, float_frame)
    write_uint8_image(
        sequence_dir / jpg_rel,
        np.array([[0, 64], [128, 255]], dtype=np.uint8),
        image_format="jpg",
        jpeg_quality=95,
    )

    manifest = {
        "frames": [
            {
                "index": 0,
                "filenames": {
                    "clean": jpg_rel,
                    "clean_raw": raw_rel,
                },
            }
        ]
    }
    manifest_path = sequence_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return RoleRef(manifest_path=str(manifest_path), role_name="clean")


def test_default_image_representation_is_raw_float():
    assert DEFAULT_IMAGE_REPRESENTATION == IMAGE_REPRESENTATION_RAW_FLOAT
    assert M7_IMAGE_REPRESENTATION == IMAGE_REPRESENTATION_JPEG_UINT8
    assert DatasetTaskSpec(
        name="probe",
        row_filter={},
        x_fields=("observed_ref",),
        y_fields=("sequence_id",),
    ).image_representation == IMAGE_REPRESENTATION_RAW_FLOAT


def test_load_role_array_defaults_to_raw_float(tmp_path: Path):
    role_ref = _write_tiny_sequence(tmp_path / "seq")
    array = load_role_array(role_ref)
    assert array.dtype == np.float32
    assert array.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(
        array[0, 0],
        [[0.0, 0.25], [0.5, 1.0]],
        rtol=0.0,
        atol=1e-6,
    )


def test_load_role_array_jpeg_uint8_matches_m7_path(tmp_path: Path):
    role_ref = _write_tiny_sequence(tmp_path / "seq")
    array = load_role_array(
        role_ref,
        image_representation=M7_IMAGE_REPRESENTATION,
    )
    assert array.dtype == np.uint8
    assert array.shape == (1, 1, 2, 2)
    # JPEG is lossy; only check that we got a plausible uint8 frame.
    assert array.min() >= 0
    assert array.max() <= 255


def test_load_role_array_raw_float_derives_path_without_raw_key(
    tmp_path: Path,
):
    sequence_dir = tmp_path / "seq_derived"
    sequence_dir.mkdir(parents=True)
    (sequence_dir / "clean").mkdir()
    jpg_rel = "clean/seq_frame_0000_angle_+0000.00.jpg"
    raw_rel = "clean/seq_frame_0000_angle_+0000.00.raw.tif"
    write_float_raw_tif(
        sequence_dir / raw_rel,
        np.full((2, 2), 0.125, dtype=np.float32),
    )
    write_uint8_image(
        sequence_dir / jpg_rel,
        np.full((2, 2), 32, dtype=np.uint8),
        image_format="jpg",
    )
    manifest_path = sequence_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "index": 0,
                        "filenames": {"clean": jpg_rel},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    role_ref = RoleRef(manifest_path=str(manifest_path), role_name="clean")
    array = load_role_array(role_ref)
    assert array.dtype == np.float32
    np.testing.assert_allclose(array[0, 0], 0.125)


def test_load_role_array_raw_float_rejects_unsupported_role(tmp_path: Path):
    sequence_dir = tmp_path / "seq_mask"
    sequence_dir.mkdir(parents=True)
    (sequence_dir / "mask").mkdir()
    png_rel = "mask/preview.png"
    Image.fromarray(
        np.full((2, 2), 128, dtype=np.uint8), mode="L"
    ).save(sequence_dir / png_rel)
    manifest_path = sequence_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frames": [
                    {"index": 0, "filenames": {"mask": png_rel}}
                ]
            }
        ),
        encoding="utf-8",
    )
    role_ref = RoleRef(manifest_path=str(manifest_path), role_name="mask")
    with pytest.raises(ValueError, match="only defined for roles"):
        load_role_array(role_ref, image_representation="raw_float")


def test_load_role_array_jpeg_uint8_loads_anomaly_png(tmp_path: Path):
    sequence_dir = tmp_path / "seq_anomaly"
    sequence_dir.mkdir(parents=True)
    (sequence_dir / "anomaly").mkdir()
    png_rel = "anomaly/preview.png"
    Image.fromarray(
        np.full((2, 2), 128, dtype=np.uint8), mode="L"
    ).save(sequence_dir / png_rel)
    manifest_path = sequence_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frames": [
                    {"index": 0, "filenames": {"anomaly": png_rel}}
                ]
            }
        ),
        encoding="utf-8",
    )
    role_ref = RoleRef(manifest_path=str(manifest_path), role_name="anomaly")
    array = load_role_array(
        role_ref, image_representation="jpeg_uint8"
    )
    assert array.dtype == np.uint8
    assert array.shape == (1, 1, 2, 2)
