"""Anomaly float raw sidecar and catalog loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from gummybear.datasets.role_images import role_image_relative_to_raw_tif
from gummybear.datasets.sequence_writer import write_sequence_roles
from tomography_ml.gummybear_data_catalog import RoleRef, load_role_array


def test_role_image_relative_to_raw_tif_accepts_png_and_jpg() -> None:
    assert (
        role_image_relative_to_raw_tif("anomaly/frame.png")
        == "anomaly/frame.raw.tif"
    )
    assert (
        role_image_relative_to_raw_tif("clean/frame.jpg") == "clean/frame.raw.tif"
    )


def test_sequence_writer_emits_anomaly_raw_delta(tmp_path: Path) -> None:
    clean = np.full((4, 4), 1.0, dtype=float)
    particle = np.full((4, 4), 1.5, dtype=float)
    manifest = {
        "frames": [{"frame_index": 0, "angle_deg": 0.0}],
        "representation": {},
    }
    result = write_sequence_roles(
        output_root=tmp_path,
        sequence_id="seq",
        angles_deg=(0.0,),
        clean_frames=(clean,),
        particle_frames=(particle,),
        manifest=manifest,
        write_anomaly_preview=True,
    )
    written = json.loads(
        (result.sequence_directory / "manifest.json").read_text(encoding="utf-8")
    )
    filenames = written["frames"][0]["filenames"]
    assert filenames["anomaly"].endswith(".png")
    assert filenames["anomaly_raw"].endswith(".raw.tif")
    assert "anomaly" in written["representation"]["raw_float_sidecar"]["roles"]

    raw_path = result.sequence_directory / filenames["anomaly_raw"]
    with Image.open(raw_path) as image:
        values = np.asarray(image, dtype=np.float32)
    # Writer flips camera-up for storage; delta is constant so flip is a no-op.
    np.testing.assert_allclose(values, 0.5, atol=1e-6)

    array = load_role_array(
        RoleRef(
            manifest_path=str(result.sequence_directory / "manifest.json"),
            role_name="anomaly",
        ),
        image_representation="raw_float",
    )
    assert array.shape == (1, 1, 4, 4)
    np.testing.assert_allclose(array[0, 0], 0.5, atol=1e-6)
