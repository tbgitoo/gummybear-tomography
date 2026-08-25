"""Milestone 8 Step 3D head-expressiveness study helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.localization import win3d_head_expressiveness_grid
from tomography_ml_validation.milestone_08.win3_study_common import (
    DEFAULT_N_REPEAT,
    load_study_results,
    mechanism_task_specs,
    run_architecture_grid_study,
    study_results_dir,
    validation_rmse_columns,
)

WIN3D_CSV = "win3d_head_expressiveness_study.csv"
WIN3D_RUNS_CSV = "win3d_head_expressiveness_runs.csv"
WIN3D_SUBDIR = "m08_3d_head_expressiveness"
WIN3D_ORDER = (
    "fourier_base_linear",
    "fourier_base_mlp",
    "flatten_base_linear",
    "flatten_base_mlp",
    "pooled_base_base",
)


def win3d_results_dir(repo_root: Path | str) -> Path:
    return study_results_dir(repo_root, WIN3D_SUBDIR)


def win3d_task_specs(**kwargs):
    return mechanism_task_specs(**kwargs)


def load_win3d_results(repo_root: Path | str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    return load_study_results(
        win3d_results_dir(repo_root),
        summary_name=WIN3D_CSV,
        runs_name=WIN3D_RUNS_CSV,
    )


def run_win3d_head_expressiveness_study(
    train_ds,
    val_ds,
    train_task,
    *,
    device,
    num_epochs: int = 200,
    batch_size: int = 16,
    early_stop_patience: int = 40,
    configs=None,
    results_dir: Path | str | None = None,
    write_csv: bool = True,
    n_repeat: int = DEFAULT_N_REPEAT,
    base_seed: int = 0,
):
    return run_architecture_grid_study(
        train_ds,
        val_ds,
        train_task,
        device=device,
        configs=configs if configs is not None else win3d_head_expressiveness_grid(),
        win="3D",
        architecture_factor="head_expressiveness",
        hypothesis="mlp_head_needed_on_frozen_fourier_primary",
        experiment_prefix="win3d",
        summary_csv_name=WIN3D_CSV,
        runs_csv_name=WIN3D_RUNS_CSV,
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        results_dir=results_dir,
        write_csv=write_csv,
        n_repeat=n_repeat,
        base_seed=base_seed,
    )


def plot_win3d_results(results_df: pd.DataFrame) -> Any:
    val_col, val_std_col = validation_rmse_columns(results_df)
    names = [name for name in WIN3D_ORDER if name in set(results_df["variant"])]
    if not names:
        names = list(results_df["variant"].astype(str))
    by_var = results_df.set_index("variant")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    means = [float(by_var.loc[name, val_col]) for name in names]
    yerr = None
    if val_std_col and val_std_col in by_var.columns:
        yerr = [float(by_var.loc[name, val_std_col]) for name in names]
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=yerr, capsize=4 if yerr else 0, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("validation RMSE total")
    ax.set_title("Milestone 8 Step 3D — head expressiveness (linear vs MLP)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def summarize_win3d(results_df: pd.DataFrame) -> list[str]:
    val_col, _ = validation_rmse_columns(results_df)
    by_var = results_df.set_index("variant")
    bullets: list[str] = []
    if {"fourier_base_linear", "fourier_base_mlp"}.issubset(by_var.index):
        lin = float(by_var.loc["fourier_base_linear", val_col])
        mlp = float(by_var.loc["fourier_base_mlp", val_col])
        bullets.append(
            f"Fourier primary: linear={lin:.3f}, MLP={mlp:.3f} "
            f"(Δ={mlp - lin:+.3f}). MLP retained for freeze."
        )
    bullets.append("Formal architecture freeze follows in Milestone 8 Step 3E — do not reopen head search in M9.")
    return bullets
