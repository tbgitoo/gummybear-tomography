"""Tests for M8 illustration dataset helpers."""

from __future__ import annotations

from pathlib import Path

from tomography_ml.gummybear_data_catalog.task_dataset import DatasetTaskSpec
from tomography_ml_validation.m8_illustration import (
    default_m8_illustration_task,
    ensure_m8_illustration_dataset,
    resolve_m8_illustration_paths,
)


def test_resolve_m8_illustration_paths_prefers_live_m8_bindings(tmp_path: Path) -> None:
    wb, out = resolve_m8_illustration_paths(tmp_path, "full")
    assert wb == tmp_path / "configs/m8/localization_single_particle.xlsx"
    assert out == tmp_path / "data/generated/m8_1/single_particle"

    live_wb = tmp_path / "configs" / "m8" / "live.xlsx"
    live_out = tmp_path / "data" / "generated" / "m8_demo"
    wb2, out2 = resolve_m8_illustration_paths(
        tmp_path,
        "full",
        workbook_path=live_wb,
        output_root=live_out,
    )
    assert wb2 == live_wb
    assert out2 == live_out


def test_ensure_reuses_existing_dataset_without_disk_io() -> None:
    sentinel_ds = object()
    task = default_m8_illustration_task()
    ds, got_task = ensure_m8_illustration_dataset(
        Path("/unused"),
        "full",
        {"dataset_M8": sentinel_ds, "localization_task_M8": task},
    )
    assert ds is sentinel_ds
    assert got_task is task


def test_default_m8_illustration_task_shape() -> None:
    task = default_m8_illustration_task()
    assert isinstance(task, DatasetTaskSpec)
    assert task.x_fields == ("anomaly_ref",)
    assert task.y_fields == ("particle_x", "particle_y", "particle_z")
    assert task.keep_angles_deg == (0.0,)
