"""Tests for DatasetTaskSpec keep_angles_deg view selection."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gummybear.datasets.role_images import write_float_raw_tif
from tomography_ml.gummybear_data_catalog import (
    CatalogRow,
    RoleRef,
    build_task_dataset,
    load_role_array,
)
from tomography_ml.gummybear_data_catalog.task_dataset import DatasetTaskSpec


def _write_multi_angle_sequence(sequence_dir: Path) -> RoleRef:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = sequence_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for index, angle in enumerate((0.0, 90.0, 180.0, 270.0)):
        jpg_rel = f"clean/seq_frame_{index:04d}_angle_{angle:+07.2f}.jpg"
        raw_rel = f"clean/seq_frame_{index:04d}_angle_{angle:+07.2f}.raw.tif"
        value = float(angle) / 270.0
        write_float_raw_tif(
            sequence_dir / raw_rel,
            np.full((2, 2), value, dtype=np.float32),
        )
        (sequence_dir / jpg_rel).write_bytes(b"not-used")
        frames.append(
            {
                "index": index,
                "angle_deg": angle,
                "filenames": {
                    "clean": jpg_rel,
                    "clean_raw": raw_rel,
                },
            }
        )

    manifest_path = sequence_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"frames": frames}),
        encoding="utf-8",
    )
    return RoleRef(manifest_path=str(manifest_path), role_name="clean")


def _catalog_row(role_ref: RoleRef, *, angles_deg: tuple[float, ...]) -> CatalogRow:
    return CatalogRow(
        sample_id=0,
        sequence_id="seq",
        split="train",
        output_root=".",
        sequence_dir=str(Path(role_ref.manifest_path).parent),
        manifest_path=role_ref.manifest_path,
        field_status="complete",
        schema_version="1.4-m6-draft",
        resolved_job_hash="x" * 64,
        camera_schedule_id="cam",
        frame_count=len(angles_deg),
        angles_deg=angles_deg,
        angles_hash="y" * 64,
        observed_ref=None,
        clean_ref=role_ref,
        particle_ref=None,
        anomaly_ref=None,
        optical_setup_id="opt_test",
        bear_mu_s=0.1,
        bear_mu_a=0.01,
        particle_present=True,
        n_particles=1,
        particle_group_id="",
        particles=(),
        particle_x=1.0,
        particle_y=2.0,
        particle_z=3.0,
        particle_radius=3.0,
        particle_mu_s=10.0,
        particle_mu_a=0.0,
        diffusion_setup_id="diff",
        extrapolation_length=1.0,
        image_domain="camera_intensity",
        composition_domain=None,
    )


def test_load_role_array_keep_single_angle(tmp_path: Path):
    role_ref = _write_multi_angle_sequence(tmp_path / "seq")
    array = load_role_array(role_ref, keep_angles_deg=180.0)
    assert array.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(array[0, 0], 180.0 / 270.0)


def test_load_role_array_keep_angle_order(tmp_path: Path):
    role_ref = _write_multi_angle_sequence(tmp_path / "seq")
    array = load_role_array(role_ref, keep_angles_deg=(180.0, 0.0))
    assert array.shape == (2, 1, 2, 2)
    np.testing.assert_allclose(array[0, 0], 180.0 / 270.0)
    np.testing.assert_allclose(array[1, 0], 0.0)


def test_build_task_dataset_filters_missing_angle(tmp_path: Path):
    role_ref = _write_multi_angle_sequence(tmp_path / "seq")
    with_180 = _catalog_row(role_ref, angles_deg=(0.0, 90.0, 180.0, 270.0))
    without_180 = replace(
        with_180,
        sample_id=1,
        sequence_id="no180",
        angles_deg=(0.0, 90.0, 270.0),
        frame_count=3,
    )
    task = DatasetTaskSpec(
        name="single_view",
        row_filter={"split": "train", "field_status": "complete"},
        x_fields=("clean_ref",),
        y_fields=("particle_x",),
        keep_angles_deg=180,
    )
    dataset = build_task_dataset([with_180, without_180], task)
    assert len(dataset) == 1
    assert dataset.rows[0].sequence_id == "seq"
    x, y = dataset[0]
    assert x["clean_ref"].shape[0] == 1
    assert y["particle_x"] == 1.0


def test_dataset_task_spec_normalizes_keep_angles():
    task = DatasetTaskSpec(
        name="probe",
        row_filter={},
        x_fields=("clean_ref",),
        y_fields=("sequence_id",),
        keep_angles_deg=180,
    )
    assert task.keep_angles_deg == (180.0,)
