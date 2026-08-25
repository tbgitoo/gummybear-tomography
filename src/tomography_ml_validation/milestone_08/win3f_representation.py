"""Milestone 8 Step 3F representation study helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.localization import win3f_representation_grid
from tomography_ml_validation.milestone_08.win3_study_common import (
    DEFAULT_N_REPEAT,
    capability_task_specs,
    load_study_results,
    run_triad_factor_study,
    study_results_dir,
    validation_rmse_columns,
)

WIN3F_CSV = "win3f_representation_study.csv"
WIN3F_RUNS_CSV = "win3f_representation_runs.csv"
WIN3F_SUBDIR = "m08_3f_representation"
WIN3F_ORDER = ("delta", "clean", "observed")
WIN3F_ARCH_ORDER = (
    "fourier_base_mlp",
    "flatten_base_mlp",
    "pooled_base_base",
)
WIN3F_ARCH_LABELS = {
    "fourier_base_mlp": "Fourier (primary)",
    "flatten_base_mlp": "Flatten (positive)",
    "pooled_base_base": "Pooled (negative)",
}


def win3f_results_dir(repo_root: Path | str) -> Path:
    return study_results_dir(repo_root, WIN3F_SUBDIR)


def load_win3f_results(repo_root: Path | str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    return load_study_results(
        win3f_results_dir(repo_root),
        summary_name=WIN3F_CSV,
        runs_name=WIN3F_RUNS_CSV,
        group_key="run_group",
    )


def run_win3f_representation_study(
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
    image_normalize: str = "per_image_zscore",
):
    factors = win3f_representation_grid()

    def build_tasks(rep_spec):
        return capability_task_specs(
            x_field=rep_spec.x_field,
            image_normalize=image_normalize,
        )

    def extra_fields(rep_spec, cfg):
        return {
            "representation": rep_spec.name,
            "x_field": rep_spec.x_field,
        }

    return run_triad_factor_study(
        catalog_rows,
        factors=factors,
        factor_column="representation",
        build_tasks_for_factor=build_tasks,
        device=device,
        win="3F",
        architecture_factor="input_representation",
        hypothesis="delta_oracle_vs_clean_vs_observed",
        experiment_prefix="win3f",
        summary_csv_name=WIN3F_CSV,
        runs_csv_name=WIN3F_RUNS_CSV,
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        results_dir=results_dir,
        write_csv=write_csv,
        n_repeat=n_repeat,
        base_seed=base_seed,
        extra_fields_for_factor=extra_fields,
    )


def plot_win3f_results(results_df: pd.DataFrame, *, runs_df: pd.DataFrame | None = None) -> Any:
    """Bar chart of validation RMSE for the Fourier primary across representations."""
    val_col, val_std_col = validation_rmse_columns(results_df)
    if "triad_role" in results_df.columns:
        primary = results_df[results_df["triad_role"] == "primary"].copy()
    else:
        primary = results_df.copy()
    if primary.empty:
        primary = results_df.copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    xs = np.arange(len(WIN3F_ORDER))
    means = []
    yerr = []
    for name in WIN3F_ORDER:
        rows = primary[primary["representation"] == name]
        if rows.empty:
            means.append(float("nan"))
            yerr.append(0.0)
            continue
        row = rows.iloc[0]
        means.append(float(row[val_col]))
        yerr.append(float(row[val_std_col]) if val_std_col and val_std_col in row else 0.0)
    ax.bar(
        xs,
        means,
        yerr=yerr if val_std_col else None,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(WIN3F_ORDER)
    ax.set_ylabel("validation RMSE total (Fourier primary)")
    ax.set_title("Milestone 8 Step 3F — representation ladder (Fourier primary)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_win3f_results_by_architecture(
    results_df: pd.DataFrame,
    *,
    runs_df: pd.DataFrame | None = None,
) -> Any:
    """Grouped bar chart: representations × all triad architectures.

    X-axis is representation (``delta`` / ``clean`` / ``observed``); one bar
    group per frozen triad architecture (Fourier / Flatten / pooled).
    """
    del runs_df  # reserved for optional per-seed overlays later
    val_col, val_std_col = validation_rmse_columns(results_df)
    if "variant" not in results_df.columns or "representation" not in results_df.columns:
        raise ValueError(
            "plot_win3f_results_by_architecture requires 'variant' and "
            "'representation' columns in results_df"
        )

    archs = [name for name in WIN3F_ARCH_ORDER if name in set(results_df["variant"].astype(str))]
    if not archs:
        archs = sorted(results_df["variant"].astype(str).unique().tolist())

    n_rep = len(WIN3F_ORDER)
    n_arch = len(archs)
    xs = np.arange(n_rep)
    width = min(0.8 / max(n_arch, 1), 0.25)
    offsets = (np.arange(n_arch) - (n_arch - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for i, arch in enumerate(archs):
        means = []
        yerr = []
        subset = results_df[results_df["variant"].astype(str) == arch]
        for rep in WIN3F_ORDER:
            rows = subset[subset["representation"] == rep]
            if rows.empty:
                means.append(float("nan"))
                yerr.append(0.0)
                continue
            row = rows.iloc[0]
            means.append(float(row[val_col]))
            yerr.append(
                float(row[val_std_col]) if val_std_col and val_std_col in row.index else 0.0
            )
        ax.bar(
            xs + offsets[i],
            means,
            width,
            yerr=yerr if val_std_col else None,
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
            label=WIN3F_ARCH_LABELS.get(arch, arch),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(WIN3F_ORDER)
    ax.set_ylabel("validation RMSE total")
    ax.set_title("Milestone 8 Step 3F — representation × architecture (triad)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def summarize_win3f(results_df: pd.DataFrame) -> list[str]:
    val_col, _ = validation_rmse_columns(results_df)
    if "triad_role" in results_df.columns:
        primary = results_df[results_df["triad_role"] == "primary"]
    else:
        primary = results_df
    bullets: list[str] = []
    for name in WIN3F_ORDER:
        rows = primary[primary["representation"] == name]
        if not rows.empty:
            bullets.append(f"{name}: mean val RMSE {float(rows.iloc[0][val_col]):.3f}.")
    bullets.append("Freeze delta (`anomaly_ref`) at 3J; observed remains operational restoration target.")
    return bullets
