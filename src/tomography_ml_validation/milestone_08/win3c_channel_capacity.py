"""Milestone 8 Step 3C channel-capacity study helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.localization import win3c_channel_capacity_grid
from tomography_ml_validation.milestone_08.win3_study_common import (
    DEFAULT_N_REPEAT,
    aggregate_runs,
    load_study_results,
    mechanism_task_specs,
    run_architecture_grid_study,
    study_results_dir,
    validation_rmse_columns,
)

WIN3C_CSV = "win3c_channel_capacity_study.csv"
WIN3C_RUNS_CSV = "win3c_channel_capacity_runs.csv"
WIN3C_SUBDIR = "m08_3c_channel_capacity"
WIN3C_ORDER = (
    "fourier_narrow_base",
    "fourier_base_base",
    "fourier_wide_base",
    "flatten_base_base",
    "pooled_base_base",
)


def win3c_results_dir(repo_root: Path | str) -> Path:
    return study_results_dir(repo_root, WIN3C_SUBDIR)


def win3c_task_specs(**kwargs):
    return mechanism_task_specs(**kwargs)


def load_win3c_results(repo_root: Path | str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    return load_study_results(
        win3c_results_dir(repo_root),
        summary_name=WIN3C_CSV,
        runs_name=WIN3C_RUNS_CSV,
    )


def run_win3c_channel_capacity_study(
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
        configs=configs if configs is not None else win3c_channel_capacity_grid(),
        win="3C",
        architecture_factor="channel_capacity",
        hypothesis="channel_width_limits_fourier_spatial_modes",
        experiment_prefix="win3c",
        summary_csv_name=WIN3C_CSV,
        runs_csv_name=WIN3C_RUNS_CSV,
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        results_dir=results_dir,
        write_csv=write_csv,
        n_repeat=n_repeat,
        base_seed=base_seed,
    )


def plot_win3c_results(
    results_df: pd.DataFrame,
    *,
    runs_df: pd.DataFrame | None = None,
) -> Any:
    val_col, val_std_col = validation_rmse_columns(results_df)
    names = [name for name in WIN3C_ORDER if name in set(results_df["variant"])]
    if not names:
        names = list(results_df["variant"].astype(str))
    by_var = results_df.set_index("variant")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    means = [float(by_var.loc[name, val_col]) for name in names]
    yerr = None
    if val_std_col and val_std_col in by_var.columns:
        yerr = [float(by_var.loc[name, val_std_col]) for name in names]
    colors = ["C2" if "fourier" in name else ("C1" if "flatten" in name else "C0") for name in names]
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=yerr, capsize=4 if yerr else 0, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("validation RMSE total")
    ax.set_title("Milestone 8 Step 3C — channel capacity (high regime, minmax)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def summarize_win3c(results_df: pd.DataFrame) -> list[str]:
    val_col, val_std_col = validation_rmse_columns(results_df)
    by_var = results_df.set_index("variant")
    bullets: list[str] = []
    fourier = [name for name in WIN3C_ORDER if name.startswith("fourier") and name in by_var.index]
    if len(fourier) >= 2:
        vals = {name: float(by_var.loc[name, val_col]) for name in fourier}
        best = min(vals, key=vals.get)
        worst = max(vals, key=vals.get)
        bullets.append(
            f"Fourier channel sweep: best={best} ({vals[best]:.3f}), "
            f"range {vals[worst] - vals[best]:.3f} over {worst}→{best}."
        )
    if "fourier_base_base" in by_var.index:
        bullets.append("Retain Fourier-base (16,32,64) as primary — wider/narrower are secondary checks.")
    if "flatten_base_base" in by_var.index:
        ref = float(by_var.loc["flatten_base_base", val_col])
        bullets.append(f"Flatten reference mean val RMSE {ref:.3f} (absolute readout upper bound).")
    if "n_repeat" in results_df.columns and int(results_df["n_repeat"].max()) > 1:
        bullets.append(
            f"Aggregated over n={int(results_df['n_repeat'].max())} seeds (mean ± std)."
        )
    return bullets
