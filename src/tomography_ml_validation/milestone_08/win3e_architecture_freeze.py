"""Milestone 8 Step 3E architecture-freeze export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tomography_ml.localization import (
    win3c_channel_capacity_grid,
    win3d_head_expressiveness_grid,
    win3e_architecture_freeze,
    win3e_control_configs,
)
from tomography_ml_validation.milestone_08.win3_study_common import (
    DEFAULT_N_REPEAT,
    load_study_results,
    mechanism_task_specs,
    run_architecture_grid_study,
    study_results_dir,
    write_json_artifact,
)
from tomography_ml_validation.milestone_08.win3c_channel_capacity import (
    WIN3C_CSV,
    WIN3C_SUBDIR,
)
from tomography_ml_validation.milestone_08.win3d_head_expressiveness import (
    WIN3D_CSV,
    WIN3D_SUBDIR,
)

WIN3E_SUBDIR = "m08_3e_architecture_freeze"
WIN3E_FREEZE_JSON = "win3e_architecture_freeze.json"
WIN3E_CONFIRM_CSV = "win3e_confirmatory_triad_study.csv"
WIN3E_CONFIRM_RUNS_CSV = "win3e_confirmatory_triad_runs.csv"


def win3e_results_dir(repo_root: Path | str) -> Path:
    return study_results_dir(repo_root, WIN3E_SUBDIR)


def write_win3e_freeze_artifacts(
    repo_root: Path | str,
    *,
    channel_capacity_df: pd.DataFrame | None = None,
    head_expressiveness_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write architecture freeze JSON and optional supporting CSV references."""
    out_dir = win3e_results_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze = win3e_architecture_freeze()
    payload = freeze.to_dict()
    payload["supporting_studies"] = {
        "win3c_relative_csv": f"{WIN3C_SUBDIR}/{WIN3C_CSV}",
        "win3d_relative_csv": f"{WIN3D_SUBDIR}/{WIN3D_CSV}",
    }
    if channel_capacity_df is not None and not channel_capacity_df.empty:
        payload["win3c_summary_rows"] = int(len(channel_capacity_df))
    if head_expressiveness_df is not None and not head_expressiveness_df.empty:
        payload["win3d_summary_rows"] = int(len(head_expressiveness_df))
    json_path = write_json_artifact(out_dir / WIN3E_FREEZE_JSON, payload)
    return {"freeze_json": json_path}


def load_win3e_supporting_results(
    repo_root: Path | str,
) -> dict[str, pd.DataFrame | None]:
    """Load local 3C / 3D summary CSVs when present."""
    root = Path(repo_root)
    c3, _ = load_study_results(
        study_results_dir(root, WIN3C_SUBDIR),
        summary_name=WIN3C_CSV,
        runs_name="win3c_channel_capacity_runs.csv",
    )
    d3, _ = load_study_results(
        study_results_dir(root, WIN3D_SUBDIR),
        summary_name=WIN3D_CSV,
        runs_name="win3d_head_expressiveness_runs.csv",
    )
    return {"3C": c3, "3D": d3}


def run_win3e_confirmatory_triad(
    train_ds,
    val_ds,
    train_task,
    *,
    device,
    num_epochs: int = 200,
    batch_size: int = 16,
    early_stop_patience: int = 40,
    results_dir: Path | str | None = None,
    write_csv: bool = True,
    n_repeat: int = DEFAULT_N_REPEAT,
    base_seed: int = 0,
):
    """Optional confirmatory train/val on the frozen interpretive triad."""
    return run_architecture_grid_study(
        train_ds,
        val_ds,
        train_task,
        device=device,
        configs=win3e_control_configs(),
        win="3E",
        architecture_factor="architecture_freeze_confirmatory",
        hypothesis="frozen_triad_reproduces_mechanism_conclusions",
        experiment_prefix="win3e",
        summary_csv_name=WIN3E_CONFIRM_CSV,
        runs_csv_name=WIN3E_CONFIRM_RUNS_CSV,
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        results_dir=results_dir,
        write_csv=write_csv,
        n_repeat=n_repeat,
        base_seed=base_seed,
    )


def win3e_task_specs(**kwargs):
    return mechanism_task_specs(**kwargs)
