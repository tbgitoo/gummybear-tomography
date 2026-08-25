"""Milestone 8 Step 3G normalisation study helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.localization import win3g_normalisation_grid
from tomography_ml_validation.milestone_08.win3_study_common import (
    DEFAULT_N_REPEAT,
    build_task_specs_with_normalisation,
    load_study_results,
    run_triad_factor_study,
    study_results_dir,
    validation_rmse_columns,
)

WIN3G_CSV = "win3g_normalisation_study.csv"
WIN3G_RUNS_CSV = "win3g_normalisation_runs.csv"
WIN3G_SUBDIR = "m08_3g_normalisation"
WIN3G_ORDER = ("raw", "train_split_zscore", "per_image_zscore", "per_image_minmax")


def win3g_results_dir(repo_root: Path | str) -> Path:
    return study_results_dir(repo_root, WIN3G_SUBDIR)


def load_win3g_results(repo_root: Path | str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    return load_study_results(
        win3g_results_dir(repo_root),
        summary_name=WIN3G_CSV,
        runs_name=WIN3G_RUNS_CSV,
        group_key="run_group",
    )


def run_win3g_normalisation_study(
    catalog_rows,
    *,
    device,
    num_epochs: int = 200,
    batch_size: int = 16,
    early_stop_patience: int = 40,
    results_dir: Path | str | None = None,
    write_csv: bool = True,
    n_repeat: int = DEFAULT_N_REPEAT,
    base_seed: int = 0,
    x_field: str = "anomaly_ref",
):
    factors = win3g_normalisation_grid()

    def build_tasks(norm_spec):
        return build_task_specs_with_normalisation(
            catalog_rows,
            x_field=x_field,
            image_normalize=norm_spec.image_normalize,
        )

    def extra_fields(norm_spec, cfg):
        return {
            "normalisation": norm_spec.name,
            "diagnostic": bool(norm_spec.diagnostic),
        }

    return run_triad_factor_study(
        catalog_rows,
        factors=factors,
        factor_column="normalisation",
        build_tasks_for_factor=build_tasks,
        device=device,
        win="3G",
        architecture_factor="intensity_normalisation",
        hypothesis="per_view_zscore_elevated_standard",
        experiment_prefix="win3g",
        summary_csv_name=WIN3G_CSV,
        runs_csv_name=WIN3G_RUNS_CSV,
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        results_dir=results_dir,
        write_csv=write_csv,
        n_repeat=n_repeat,
        base_seed=base_seed,
        extra_fields_for_factor=extra_fields,
    )


def plot_win3g_results(results_df: pd.DataFrame) -> Any:
    if results_df is None or results_df.empty:
        raise ValueError(
            "No Milestone 8 Step 3G results to plot — run the train/validation "
            "study cell first."
        )
    val_col, val_std_col = validation_rmse_columns(results_df)
    if "triad_role" in results_df.columns:
        primary = results_df[results_df["triad_role"] == "primary"]
    else:
        primary = results_df
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    xs = np.arange(len(WIN3G_ORDER))
    means, yerr = [], []
    for name in WIN3G_ORDER:
        rows = primary[primary["normalisation"] == name]
        if rows.empty:
            means.append(float("nan"))
            yerr.append(0.0)
            continue
        row = rows.iloc[0]
        means.append(float(row[val_col]))
        yerr.append(float(row[val_std_col]) if val_std_col and val_std_col in row else 0.0)
    ax.bar(xs, means, yerr=yerr if val_std_col else None, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(WIN3G_ORDER, rotation=20, ha="right")
    ax.set_ylabel("validation RMSE total (Fourier primary)")
    ax.set_title("Milestone 8 Step 3G — normalisation ladder")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def summarize_win3g(results_df: pd.DataFrame) -> list[str]:
    if results_df is None or results_df.empty:
        return ["No Milestone 8 Step 3G results yet — finish the train/validation study cell."]
    val_col, _ = validation_rmse_columns(results_df)
    if "triad_role" in results_df.columns:
        primary = results_df[results_df["triad_role"] == "primary"]
    else:
        primary = results_df
    bullets: list[str] = []
    for name in WIN3G_ORDER:
        rows = primary[primary["normalisation"] == name]
        if not rows.empty:
            bullets.append(f"{name}: mean val RMSE {float(rows.iloc[0][val_col]):.3f}.")
    bullets.append("Elevated standard: per_image_zscore (per-view z-score) for downstream M9/M10.")
    return bullets
