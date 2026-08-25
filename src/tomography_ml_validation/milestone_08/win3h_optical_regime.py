"""Milestone 8 Step 3H optical-regime study helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.localization import win3h_optical_regime_grid
from tomography_ml_validation.milestone_08.win3_study_common import (
    DEFAULT_N_REPEAT,
    capability_task_specs,
    load_study_results,
    run_triad_factor_study,
    study_results_dir,
    validation_rmse_columns,
)

WIN3H_CSV = "win3h_optical_regime_study.csv"
WIN3H_RUNS_CSV = "win3h_optical_regime_runs.csv"
WIN3H_SUBDIR = "m08_3h_optical_regime"
WIN3H_ORDER = ("low", "medium", "high")


def win3h_results_dir(repo_root: Path | str) -> Path:
    return study_results_dir(repo_root, WIN3H_SUBDIR)


def load_win3h_results(repo_root: Path | str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    return load_study_results(
        win3h_results_dir(repo_root),
        summary_name=WIN3H_CSV,
        runs_name=WIN3H_RUNS_CSV,
        group_key="run_group",
    )


def run_win3h_optical_regime_study(
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
    image_normalize: str = "per_image_zscore",
    resume: bool = True,
):
    factors = win3h_optical_regime_grid()

    def build_tasks(regime_spec):
        return capability_task_specs(
            x_field=x_field,
            image_normalize=image_normalize,
            optical_setup_id=regime_spec.optical_setup_id,
        )

    def extra_fields(regime_spec, cfg):
        return {
            "optical_regime": regime_spec.name,
            "optical_setup_id": regime_spec.optical_setup_id,
            "mu_s": regime_spec.mu_s,
            "mu_a": regime_spec.mu_a,
        }

    return run_triad_factor_study(
        catalog_rows,
        factors=factors,
        factor_column="optical_regime",
        build_tasks_for_factor=build_tasks,
        device=device,
        win="3H",
        architecture_factor="optical_regime",
        hypothesis="background_optics_nuisance_not_model_input",
        experiment_prefix="win3h",
        summary_csv_name=WIN3H_CSV,
        runs_csv_name=WIN3H_RUNS_CSV,
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        results_dir=results_dir,
        write_csv=write_csv,
        n_repeat=n_repeat,
        base_seed=base_seed,
        resume=resume,
        extra_fields_for_factor=extra_fields,
    )


def plot_win3h_results(results_df: pd.DataFrame) -> Any:
    if results_df is None or results_df.empty:
        raise ValueError(
            "No Milestone 8 Step 3H results to plot — run the train/validation "
            "study cell first (or set RUN_STUDY=True)."
        )
    val_col, val_std_col = validation_rmse_columns(results_df)
    if "triad_role" in results_df.columns:
        primary = results_df[results_df["triad_role"] == "primary"]
    else:
        primary = results_df
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(WIN3H_ORDER))
    means, yerr = [], []
    for name in WIN3H_ORDER:
        rows = primary[primary["optical_regime"] == name]
        if rows.empty:
            means.append(float("nan"))
            yerr.append(0.0)
            continue
        row = rows.iloc[0]
        means.append(float(row[val_col]))
        yerr.append(float(row[val_std_col]) if val_std_col and val_std_col in row else 0.0)
    ax.bar(xs, means, yerr=yerr if val_std_col else None, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(WIN3H_ORDER)
    ax.set_ylabel("validation RMSE total (Fourier primary)")
    ax.set_title("Milestone 8 Step 3H — optical regime stratification")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def summarize_win3h(results_df: pd.DataFrame) -> list[str]:
    if results_df is None or results_df.empty:
        return [
            "No Milestone 8 Step 3H results yet — finish the train/validation study cell "
            "(interrupted runs resume from partial CSVs under checkpoints/m8/m08_3h_optical_regime/)."
        ]
    val_col, _ = validation_rmse_columns(results_df)
    if "triad_role" in results_df.columns:
        primary = results_df[results_df["triad_role"] == "primary"]
    else:
        primary = results_df
    bullets: list[str] = []
    for name in WIN3H_ORDER:
        rows = primary[primary["optical_regime"] == name]
        if not rows.empty:
            bullets.append(f"{name}: mean val RMSE {float(rows.iloc[0][val_col]):.3f}.")
    bullets.append("Report ML in Final Report filters high regime only; regimes are nuisance stratification.")
    return bullets
