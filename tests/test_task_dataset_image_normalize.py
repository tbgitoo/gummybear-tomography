"""Tests for DatasetTaskSpec image_normalize modes (WIN 3G)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gummybear.datasets.role_images import write_float_raw_tif
from tomography_ml.gummybear_data_catalog import RoleRef, build_task_dataset
from tomography_ml.gummybear_data_catalog.catalog import CatalogRow
from tomography_ml.gummybear_data_catalog.task_dataset import (
    IMAGE_NORMALIZE_PER_IMAGE_MINMAX,
    IMAGE_NORMALIZE_PER_IMAGE_ZSCORE,
    IMAGE_NORMALIZE_TRAIN_SPLIT_ZSCORE,
    DatasetTaskSpec,
    IntensityStats,
    apply_image_normalize,
    estimate_intensity_stats,
    resolve_task_sample,
)


def test_apply_per_image_minmax_maps_each_view_to_unit_interval() -> None:
    array = np.zeros((2, 1, 2, 2), dtype=np.float32)
    array[0] = 10.0
    array[1, 0] = np.array([[-2.0, 2.0], [0.0, 1.0]], dtype=np.float32)

    out = apply_image_normalize(array, IMAGE_NORMALIZE_PER_IMAGE_MINMAX)
    assert out.shape == array.shape
    np.testing.assert_allclose(out[0], 0.0)  # constant -> zeros
    np.testing.assert_allclose(out[1, 0].min(), 0.0)
    np.testing.assert_allclose(out[1, 0].max(), 1.0)
    np.testing.assert_allclose(out[1, 0, 0, 0], 0.0)  # -2 -> 0
    np.testing.assert_allclose(out[1, 0, 0, 1], 1.0)  # 2 -> 1


def test_apply_per_image_zscore_zero_mean_unit_std() -> None:
    array = np.zeros((1, 1, 2, 2), dtype=np.float32)
    array[0, 0] = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    out = apply_image_normalize(array, IMAGE_NORMALIZE_PER_IMAGE_ZSCORE)
    np.testing.assert_allclose(out.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(out.std(), 1.0, atol=1e-6)

    constant = np.ones((1, 1, 2, 2), dtype=np.float32)
    out_c = apply_image_normalize(constant, IMAGE_NORMALIZE_PER_IMAGE_ZSCORE)
    np.testing.assert_allclose(out_c, 0.0)


def test_apply_train_split_zscore_uses_provided_stats() -> None:
    array = np.array([[[[1.0, 3.0], [5.0, 7.0]]]], dtype=np.float32)
    stats = IntensityStats(mean=4.0, std=2.0)
    out = apply_image_normalize(
        array,
        IMAGE_NORMALIZE_TRAIN_SPLIT_ZSCORE,
        intensity_stats=stats,
    )
    expected = (array - 4.0) / 2.0
    np.testing.assert_allclose(out, expected)
    with pytest.raises(ValueError, match="intensity_stats"):
        apply_image_normalize(array, IMAGE_NORMALIZE_TRAIN_SPLIT_ZSCORE)


def _make_anomaly_row(tmp_path: Path, values: np.ndarray) -> CatalogRow:
    sequence_dir = tmp_path / "seq"
    anomaly_dir = sequence_dir / "anomaly"
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    raw_rel = "anomaly/frame.raw.tif"
    write_float_raw_tif(sequence_dir / raw_rel, values)
    manifest_path = sequence_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "index": 0,
                        "angle_deg": 0.0,
                        "filenames": {
                            "anomaly": "anomaly/frame.png",
                            "anomaly_raw": raw_rel,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    role_ref = RoleRef(manifest_path=str(manifest_path), role_name="anomaly")
    return CatalogRow(
        sample_id=0,
        sequence_id="seq",
        split="train",
        output_root=str(tmp_path),
        sequence_dir=str(sequence_dir),
        manifest_path=str(manifest_path),
        field_status="complete",
        schema_version="1.5-m6-draft",
        resolved_job_hash="a" * 64,
        camera_schedule_id="cam",
        frame_count=1,
        angles_deg=(0.0,),
        angles_hash="b" * 64,
        observed_ref=None,
        clean_ref=None,
        particle_ref=None,
        anomaly_ref=role_ref,
        optical_setup_id="opt",
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


def test_resolve_task_sample_applies_minmax_to_x_role(
    tmp_path: Path,
) -> None:
    values = np.array([[1.0, 5.0], [3.0, 9.0]], dtype=np.float32)
    row = _make_anomaly_row(tmp_path, values)
    task = DatasetTaskSpec(
        name="anomaly_norm",
        row_filter={"split": "train", "field_status": "complete"},
        x_fields=("anomaly_ref",),
        y_fields=("particle_x",),
        image_normalize="per_image_minmax",
        keep_angles_deg=0,
    )
    x, y = resolve_task_sample(row, task)
    arr = x["anomaly_ref"]
    assert arr.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(arr.min(), 0.0)
    np.testing.assert_allclose(arr.max(), 1.0)
    # (1-1)/(9-1)=0, (9-1)/(9-1)=1
    np.testing.assert_allclose(arr[0, 0, 0, 0], 0.0)
    np.testing.assert_allclose(arr[0, 0, 1, 1], 1.0)
    assert y["particle_x"] == 1.0

    dataset = build_task_dataset([row], task)
    x2, _ = dataset[0]
    np.testing.assert_allclose(x2["anomaly_ref"], arr)


def test_estimate_intensity_stats_and_train_split_task(tmp_path: Path) -> None:
    values = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    row = _make_anomaly_row(tmp_path, values)
    raw_task = DatasetTaskSpec(
        name="raw",
        row_filter={"split": "train", "field_status": "complete"},
        x_fields=("anomaly_ref",),
        y_fields=("particle_x",),
        image_normalize="none",
        keep_angles_deg=0,
    )
    raw_ds = build_task_dataset([row], raw_task)
    stats = estimate_intensity_stats(raw_ds, "anomaly_ref")
    np.testing.assert_allclose(stats.mean, 4.0)
    np.testing.assert_allclose(stats.std, np.sqrt(5.0))  # pop std of 1,3,5,7

    z_task = DatasetTaskSpec(
        name="z",
        row_filter={"split": "train", "field_status": "complete"},
        x_fields=("anomaly_ref",),
        y_fields=("particle_x",),
        image_normalize="train_split_zscore",
        intensity_mean=stats.mean,
        intensity_std=stats.std,
        keep_angles_deg=0,
    )
    x, _ = resolve_task_sample(row, z_task)
    expected = (values - stats.mean) / stats.std
    np.testing.assert_allclose(x["anomaly_ref"][0, 0], expected, rtol=1e-5)


def test_train_split_zscore_requires_stats_on_spec() -> None:
    with pytest.raises(ValueError, match="intensity_mean"):
        DatasetTaskSpec(
            name="bad",
            row_filter={},
            x_fields=("anomaly_ref",),
            y_fields=("particle_x",),
            image_normalize="train_split_zscore",
        )
    with pytest.raises(ValueError, match="only valid"):
        DatasetTaskSpec(
            name="bad2",
            row_filter={},
            x_fields=("anomaly_ref",),
            y_fields=("particle_x",),
            image_normalize="none",
            intensity_mean=0.0,
            intensity_std=1.0,
        )
