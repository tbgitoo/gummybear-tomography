"""Milestone 8 Step 3I observability consolidation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.localization import win3i_key_result_sources, win3j_single_view_freeze
from tomography_ml.gummybear_data_catalog.task_dataset import build_task_dataset
from tomography_ml_validation.milestone_08.notebook_helpers import load_historical_win3_csv
from tomography_ml_validation.milestone_08.win3_study_common import (
    DEFAULT_N_REPEAT,
    capability_task_specs,
    load_study_results,
    run_architecture_grid_study,
    study_results_dir,
    write_json_artifact,
)
from tomography_ml.localization import win3e_control_configs

WIN3I_SUBDIR = "m08_3i_observability"
WIN3I_SUMMARY_CSV = "win3i_observability_summary.csv"
WIN3I_CONFIRM_CSV = "win3i_confirmatory_triad_study.csv"
WIN3I_CONFIRM_RUNS_CSV = "win3i_confirmatory_triad_runs.csv"
WIN3I_MANIFEST_JSON = "win3i_observability_manifest.json"
# Keys match evaluate_split_rmse / per_axis_rmse (uppercase axis letters).
AXIS_METRICS = (
    ("validation_RMSE_X", "X"),
    ("validation_RMSE_Y", "Y"),
    ("validation_RMSE_Z", "Z"),
)


def win3i_results_dir(repo_root: Path | str) -> Path:
    return study_results_dir(repo_root, WIN3I_SUBDIR)


def load_win3i_source_tables(repo_root: Path | str) -> dict[str, pd.DataFrame | None]:
    """Load 3F / 3G / 3H summary CSVs from the local m8_1 results tree."""
    loaded: dict[str, pd.DataFrame | None] = {}
    for src in win3i_key_result_sources():
        loaded[src["win"]] = load_historical_win3_csv(repo_root, src["relative_csv"])
    return loaded


def consolidate_win3i_axis_table(
    source_tables: Mapping[str, pd.DataFrame | None],
) -> pd.DataFrame:
    """Flatten per-axis validation RMSE from capability-study summaries."""
    rows: list[dict[str, Any]] = []
    for win, df in source_tables.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            payload = row.to_dict()
            payload["source_win"] = win
            rows.append(payload)
    return pd.DataFrame(rows)


def run_win3i_confirmatory_study(
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
):
    """Confirmatory triad at the elevated per-view z-score standard."""
    block = win3j_single_view_freeze()
    train_task, val_task, test_task = capability_task_specs(
        x_field=block.x_field,
        image_normalize=block.image_normalize,
        optical_setup_id=block.optical_setup_id_reference,
    )
    train_ds = build_task_dataset(catalog_rows, train_task)
    val_ds = build_task_dataset(catalog_rows, val_task)
    return run_architecture_grid_study(
        train_ds,
        val_ds,
        train_task,
        device=device,
        configs=win3e_control_configs(),
        win="3I",
        architecture_factor="observability_confirmatory",
        hypothesis="report_total_and_per_axis_rmse",
        experiment_prefix="win3i",
        summary_csv_name=WIN3I_CONFIRM_CSV,
        runs_csv_name=WIN3I_CONFIRM_RUNS_CSV,
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        results_dir=results_dir,
        write_csv=write_csv,
        n_repeat=n_repeat,
        base_seed=base_seed,
    )


def write_win3i_manifest(
    repo_root: Path | str,
    *,
    source_tables: Mapping[str, pd.DataFrame | None],
    confirmatory_df: pd.DataFrame | None = None,
) -> Path:
    out_dir = win3i_results_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sources": [dict(src) for src in win3i_key_result_sources()],
        "loaded_rows": {
            win: int(len(df)) if df is not None else 0
            for win, df in source_tables.items()
        },
        "confirmatory_rows": int(len(confirmatory_df)) if confirmatory_df is not None else 0,
    }
    return write_json_artifact(out_dir / WIN3I_MANIFEST_JSON, payload)


def load_win3i_confirmatory_results(
    repo_root: Path | str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    return load_study_results(
        win3i_results_dir(repo_root),
        summary_name=WIN3I_CONFIRM_CSV,
        runs_name=WIN3I_CONFIRM_RUNS_CSV,
    )


def _axis_metric_value(row: pd.Series, metric: str) -> tuple[float, float | None]:
    """Return (mean, optional std) for an axis metric on a summary or run row."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col in row.index and pd.notna(row[mean_col]):
        std = float(row[std_col]) if std_col in row.index and pd.notna(row[std_col]) else None
        return float(row[mean_col]), std
    if metric in row.index and pd.notna(row[metric]):
        return float(row[metric]), None
    return float("nan"), None


def plot_win3i_per_axis(
    results_df: pd.DataFrame,
    *,
    variant: str | None = None,
) -> Any:
    """Bar chart of validation RMSE per axis for one variant (default Fourier primary)."""
    if results_df is None or results_df.empty:
        raise ValueError(
            "No Milestone 8 Step 3I confirmatory results to plot — run the study cell first."
        )
    if variant is None:
        variant = "fourier_base_mlp"
    rows = results_df[results_df["variant"] == variant]
    if rows.empty:
        rows = results_df.iloc[[0]]
        variant = str(rows.iloc[0].get("variant", variant))
    row = rows.iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    labels = [label for _, label in AXIS_METRICS]
    means: list[float] = []
    yerr: list[float] = []
    for metric, _ in AXIS_METRICS:
        val, std = _axis_metric_value(row, metric)
        means.append(val)
        yerr.append(0.0 if std is None else std)
    if all(np.isnan(means)):
        raise ValueError(
            "Per-axis validation RMSE columns missing "
            f"(expected {[m for m, _ in AXIS_METRICS]} or *_mean). "
            "Re-run the confirmatory study or reload runs CSV after upgrading helpers."
        )
    xs = np.arange(len(labels))
    use_err = any(e > 0 for e in yerr)
    ax.bar(
        xs,
        means,
        yerr=yerr if use_err else None,
        capsize=4 if use_err else 0,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("validation RMSE")
    ax.set_title(f"Milestone 8 Step 3I — per-axis RMSE ({variant})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def summarize_win3i(
    source_tables: Mapping[str, pd.DataFrame | None],
    confirmatory_df: pd.DataFrame | None = None,
) -> list[str]:
    bullets: list[str] = []
    for win, df in source_tables.items():
        if df is None or df.empty:
            bullets.append(f"M8 Step {win}: no local CSV yet — run the corresponding study notebook.")
        else:
            bullets.append(f"M8 Step {win}: loaded {len(df)} summary rows.")
    if confirmatory_df is not None and not confirmatory_df.empty:
        bullets.append(
            "Confirmatory triad loaded — report validation RMSE total **and** per-axis X/Y/Z."
        )
    bullets.append("Single-view axis difficulty is not isotropic; do not hide structure in one scalar.")
    return bullets
