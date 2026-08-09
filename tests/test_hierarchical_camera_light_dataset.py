"""Tests for HierarchicalCameraLightDataset (standalone x,y indexing)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from tomography_ml.gummybear_data_catalog import HierarchicalCameraLightDataset


def _row(*, seq: str, oid: str = "opt_m10_illum_000"):
    return SimpleNamespace(sequence_id=seq, optical_setup_id=oid)


def test_hierarchical_dataset_xy_contract_and_shape(monkeypatch) -> None:
    lights = (0.0, 60.0)
    cams = (0.0, 90.0)
    row0 = _row(seq="bear_9_a", oid="opt_m10_illum_000")
    row1 = _row(seq="bear_9_b", oid="opt_m10_illum_060")
    group = {
        "particle_id": "9",
        "split": "train",
        "rows_by_light": {0.0: row0, 60.0: row1},
    }

    def fake_resolve(row, task):
        # Per light: V=2 cameras, C=1, H=W=4
        val = 1.0 if row is row0 else 2.0
        images = {"anomaly_ref": np.full((2, 1, 4, 4), val, dtype=np.float32)}
        y = {"particle_x": 1.0, "particle_y": 2.0, "particle_z": 3.0}
        return images, y

    mod = importlib.import_module(
        "tomography_ml.gummybear_data_catalog.HierarchicalCameraLightDataset"
    )
    monkeypatch.setattr(mod, "resolve_task_sample", fake_resolve)

    ds = HierarchicalCameraLightDataset(
        [group],
        x_field="anomaly_ref",
        y_fields=("particle_x", "particle_y", "particle_z"),
        camera_angles_deg=cams,
        light_angles_deg=lights,
        image_normalize="none",
    )
    assert len(ds) == 1
    assert HierarchicalCameraLightDataset.__bases__ == (object,)

    x, y = ds[0]
    assert set(x) == {"anomaly_ref"}
    views = x["anomaly_ref"]
    assert isinstance(views, np.ndarray)
    assert views.shape == (2, 2, 1, 4, 4)  # [I, V, C, H, W]
    assert float(views[0].mean()) == 1.0
    assert float(views[1].mean()) == 2.0
    assert y == {"particle_x": 1.0, "particle_y": 2.0, "particle_z": 3.0}
    assert ds.camera_angles_deg == cams
    assert ds.light_angles_deg == lights


def test_hierarchical_dataset_rejects_empty_schedules() -> None:
    with pytest.raises(ValueError, match="camera_angles_deg"):
        HierarchicalCameraLightDataset(
            [],
            x_field="anomaly_ref",
            y_fields=("particle_x",),
            camera_angles_deg=(),
            light_angles_deg=(0.0,),
        )
    with pytest.raises(ValueError, match="light_angles_deg"):
        HierarchicalCameraLightDataset(
            [],
            x_field="anomaly_ref",
            y_fields=("particle_x",),
            camera_angles_deg=(0.0,),
            light_angles_deg=(),
        )


def test_hierarchical_dataset_view_count_mismatch(monkeypatch) -> None:
    row0 = _row(seq="bear_1")
    group = {
        "particle_id": "1",
        "split": "train",
        "rows_by_light": {0.0: row0},
    }

    def fake_resolve(row, task):
        images = {"anomaly_ref": np.zeros((1, 1, 2, 2), dtype=np.float32)}
        return images, {"particle_x": 0.0}

    mod = importlib.import_module(
        "tomography_ml.gummybear_data_catalog.HierarchicalCameraLightDataset"
    )
    monkeypatch.setattr(mod, "resolve_task_sample", fake_resolve)
    ds = HierarchicalCameraLightDataset(
        [group],
        x_field="anomaly_ref",
        y_fields=("particle_x",),
        camera_angles_deg=(0.0, 90.0),
        light_angles_deg=(0.0,),
    )
    with pytest.raises(ValueError, match="expected 2 camera views"):
        _ = ds[0]
