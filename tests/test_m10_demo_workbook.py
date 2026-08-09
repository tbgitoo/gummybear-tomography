"""Tests for the hand-maintained M10 demo workbook fixture."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import pandas as pd

from gummybear.datasets.generation_plan import validate_generation_plan
from gummybear.datasets.generation_workbook import load_generation_workbook
from tomography_ml.localization.localize_multiview import (
    light_angle_deg_from_optical_setup_id,
)

_PACKAGED_M10_DEMO = (
    files("tomography_ml_validation")
    / "test_data"
    / "configs"
    / "m10"
    / "m10_demo.xlsx"
)


def test_packaged_m10_demo_workbook_shape_and_splits():
    with as_file(_PACKAGED_M10_DEMO) as path:
        frames = pd.read_excel(path, sheet_name=None, engine="openpyxl")

    assert len(frames["optical_setups"]) == 3
    assert len(frames["particles"]) == 8
    assert len(frames["sequences"]) == 24  # 8 particles × 3 lights
    assert int(frames["camera_schedules"].iloc[0]["num_views"]) == 4

    sequences = frames["sequences"]
    assert (sequences["split"] == "train").sum() == 12
    assert (sequences["split"] == "validation").sum() == 6
    assert (sequences["split"] == "test").sum() == 6
    assert set(sequences["output_root"].unique()) == {"data/generated/m10_demo"}

    light_angles = {
        float(light_angle_deg_from_optical_setup_id(str(oid)))
        for oid in sequences["optical_setup_id"].unique()
    }
    assert light_angles == {0.0, 120.0, 240.0}


def test_packaged_m10_demo_workbook_validates():
    with as_file(_PACKAGED_M10_DEMO) as path:
        workbook = load_generation_workbook(path)
    repo = Path(__file__).resolve().parents[1]
    plan = validate_generation_plan(workbook, repo_root=repo, stl_root=repo)
    assert len(plan.jobs) == 24
    assert all(len(job.camera.poses) == 4 for job in plan.jobs)
