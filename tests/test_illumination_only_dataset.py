"""Tests for illumination-only joint groups + dataset helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch

from tomography_ml.gummybear_data_catalog import (
    IlluminationOnlyDataset,
    build_illumination_joint_groups,
    count_groups_by_split,
    groups_for_split,
    particle_id_from_sequence_id,
)


def test_particle_id_from_sequence_id() -> None:
    assert particle_id_from_sequence_id("bear_m10_000_000042") == "000042"
    assert particle_id_from_sequence_id("x_y") == "y"


def _row(*, seq: str, oid: str, split: str = "train", status: str = "complete"):
    return SimpleNamespace(
        sequence_id=seq,
        optical_setup_id=oid,
        split=split,
        field_status=status,
    )


def test_build_illumination_joint_groups_filters_incomplete_lights() -> None:
    lights = (0.0, 60.0, 120.0)
    rows = [
        _row(seq="bear_1", oid="opt_m10_illum_000", split="train"),
        _row(seq="bear_1", oid="opt_m10_illum_060", split="train"),
        _row(seq="bear_1", oid="opt_m10_illum_120", split="train"),
        # particle 2: missing 120
        _row(seq="bear_2", oid="opt_m10_illum_000", split="train"),
        _row(seq="bear_2", oid="opt_m10_illum_060", split="train"),
    ]
    groups = build_illumination_joint_groups(rows, light_angles_deg=lights)
    assert len(groups) == 1
    assert groups[0]["particle_id"] == "1"
    assert set(groups[0]["rows_by_light"]) == set(lights)
    assert count_groups_by_split(groups) == {"train": 1}
    assert len(groups_for_split(groups, "validation")) == 0


def test_build_illumination_joint_groups_min_groups() -> None:
    lights = (0.0,)
    rows = [_row(seq="bear_1", oid="opt_m10_illum_000")]
    try:
        build_illumination_joint_groups(rows, light_angles_deg=lights, min_groups=4)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "min_groups=4" in str(exc)


def test_illumination_only_dataset_stacks_views(monkeypatch) -> None:
    lights = (0.0, 60.0)
    row0 = _row(seq="bear_9", oid="opt_m10_illum_000")
    row1 = _row(seq="bear_9", oid="opt_m10_illum_060")
    group = {
        "particle_id": "9",
        "split": "train",
        "rows_by_light": {0.0: row0, 60.0: row1},
    }

    def fake_resolve(row, task):
        val = 1.0 if row is row0 else 2.0
        images = {"anomaly_ref": torch.full((1, 4, 4), val)}
        y = {"particle_x": 1.0, "particle_y": 2.0, "particle_z": 3.0}
        return images, y

    # Module file shares the class name; load submodule explicitly for patching.
    iod_mod = importlib.import_module(
        "tomography_ml.gummybear_data_catalog.IlluminationOnlyDataset"
    )
    monkeypatch.setattr(iod_mod, "resolve_task_sample", fake_resolve)
    ds = IlluminationOnlyDataset(
        [group],
        x_field="anomaly_ref",
        y_fields=("particle_x", "particle_y", "particle_z"),
        fixed_camera_deg=180.0,
        light_angles_deg=lights,
        image_normalize="none",
    )
    assert len(ds) == 1
    views, targets, angles = ds[0]
    assert views.shape == (2, 1, 4, 4)
    assert float(views[0].mean()) == 1.0
    assert float(views[1].mean()) == 2.0
    assert targets == {"particle_x": 1.0, "particle_y": 2.0, "particle_z": 3.0}
    assert torch.allclose(angles, torch.tensor([0.0, 60.0]))
