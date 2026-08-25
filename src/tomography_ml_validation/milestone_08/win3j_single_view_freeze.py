"""Milestone 8 Step 3J single-view block freeze study helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.localization import win3e_control_configs, win3j_single_view_freeze
from tomography_ml.gummybear_data_catalog.task_dataset import build_task_dataset
from tomography_ml_validation.milestone_08.win3_study_common import (
    DEFAULT_N_REPEAT,
    capability_task_specs,
    load_study_results,
    run_architecture_grid_study,
    study_results_dir,
    validation_rmse_columns,
    write_json_artifact,
)

WIN3J_SUBDIR = "m08_3j_single_view_freeze"
WIN3J_STUDY_CSV = "win3j_single_view_freeze_study.csv"
WIN3J_RUNS_CSV = "win3j_single_view_freeze_runs.csv"
WIN3J_FREEZE_JSON = "win3j_single_view_freeze.json"


def win3j_results_dir(repo_root: Path | str) -> Path:
    return study_results_dir(repo_root, WIN3J_SUBDIR)


def win3j_task_specs():
    block = win3j_single_view_freeze()
    return capability_task_specs(
        x_field=block.x_field,
        image_normalize=block.image_normalize,
        optical_setup_id=block.optical_setup_id_reference,
    )


def load_win3j_results(repo_root: Path | str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    return load_study_results(
        win3j_results_dir(repo_root),
        summary_name=WIN3J_STUDY_CSV,
        runs_name=WIN3J_RUNS_CSV,
    )


def run_win3j_freeze_study(
    catalog_rows,
    *,
    device,
    num_epochs: int = 200,
    batch_size: int = 16,
    early_stop_patience: int = 40,
    results_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
    write_csv: bool = True,
    n_repeat: int = 1,
    base_seed: int = 0,
    triad: bool = True,
):
    """One-shot train/val/test on the frozen interpretive triad (default) or primary only."""
    train_task, val_task, test_task = win3j_task_specs()
    block = win3j_single_view_freeze()
    batch_size = batch_size or block.batch_size
    early_stop_patience = early_stop_patience or block.early_stop_patience
    num_epochs = num_epochs or block.num_epochs_max
    train_ds = build_task_dataset(catalog_rows, train_task)
    val_ds = build_task_dataset(catalog_rows, val_task)
    test_ds = build_task_dataset(catalog_rows, test_task)
    configs = win3e_control_configs() if triad else (block.architecture.primary_config(),)
    out_dir = Path(results_dir) if results_dir is not None else None
    summary_df, runs_df, histories = run_architecture_grid_study(
        train_ds,
        val_ds,
        train_task,
        device=device,
        configs=configs,
        win="3J",
        architecture_factor="single_view_block_freeze",
        hypothesis="freeze_before_multi_view_fusion",
        experiment_prefix="win3j",
        summary_csv_name=WIN3J_STUDY_CSV,
        runs_csv_name=WIN3J_RUNS_CSV,
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        results_dir=out_dir,
        write_csv=write_csv,
        n_repeat=n_repeat,
        base_seed=base_seed,
        test_ds=test_ds,
        evaluate_test=True,
    )
    if write_csv:
        if repo_root is not None:
            root = Path(repo_root)
        elif out_dir is not None:
            # …/checkpoints/m8/<study> → repo root is parents[2]
            root = out_dir.parents[2]
        else:
            root = Path(".")
        write_win3j_freeze_json(root, runs_df=runs_df)
    return summary_df, runs_df, histories


def write_win3j_freeze_json(
    repo_root: Path | str,
    *,
    runs_df: pd.DataFrame | None = None,
) -> Path:
    out_dir = win3j_results_dir(repo_root)
    block = win3j_single_view_freeze()
    payload = block.to_dict()
    if runs_df is not None and not runs_df.empty:
        payload["study_rows"] = int(len(runs_df))
        payload["variants"] = sorted(runs_df["variant"].astype(str).unique().tolist())
    return write_json_artifact(out_dir / WIN3J_FREEZE_JSON, payload)


def plot_win3j_results(results_df: pd.DataFrame) -> Any:
    val_col, _ = validation_rmse_columns(results_df)
    test_col = "test_RMSE_total_mean" if "test_RMSE_total_mean" in results_df.columns else "test_RMSE_total"
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    names = list(results_df["variant"].astype(str))
    xs = np.arange(len(names))
    val_means = [float(v) for v in results_df[val_col]]
    test_means = [float(v) for v in results_df[test_col]]
    width = 0.35
    ax.bar(xs - width / 2, val_means, width, label="validation", edgecolor="black", linewidth=0.5)
    ax.bar(xs + width / 2, test_means, width, label="test", edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("RMSE total")
    ax.set_title("Milestone 8 Step 3J — frozen triad train/val/test")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def summarize_win3j(results_df: pd.DataFrame) -> list[str]:
    val_col, _ = validation_rmse_columns(results_df)
    test_col = "test_RMSE_total_mean" if "test_RMSE_total_mean" in results_df.columns else "test_RMSE_total"
    bullets: list[str] = []
    for _, row in results_df.iterrows():
        bullets.append(
            f"{row['variant']}: val={float(row[val_col]):.3f}, test={float(row[test_col]):.3f}."
        )
    bullets.append(
        "Freeze record written to `checkpoints/m8/m08_3j_single_view_freeze/` — "
        "do not silently retune for M9."
    )
    return bullets
