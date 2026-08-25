"""Milestone 8 Step 3B receptive-field study helpers for milestone_08 notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.localization import (
    SingleViewArchConfig,
    win3b_receptive_field_grid,
)
from tomography_ml_validation.milestone_08.win3_study_common import (
    DEFAULT_N_REPEAT,
    aggregate_runs,
    load_study_results,
    lr_for_config,
    mechanism_task_specs,
    run_architecture_grid_study,
    study_results_dir,
    validation_rmse_columns,
)

WIN3B_CSV_REL = "m08_3b_receptive_field/win3b_receptive_field_study.csv"
WIN3B_RUNS_CSV_REL = "m08_3b_receptive_field/win3b_receptive_field_runs.csv"
WIN3B_CSV = "win3b_receptive_field_study.csv"
WIN3B_RUNS_CSV = "win3b_receptive_field_runs.csv"
WIN3B_SUBDIR = "m08_3b_receptive_field"
WIN3B_DEPTH_ORDER = (
    "fourier_shallow_base",
    "fourier_base_base",
    "fourier_deeper_base",
)
WIN3B_DOWNSAMPLE_ORDER = (
    "fourier_base_base",
    "fourier_base_low",
    "fourier_base_medium",
    "fourier_base_high",
)
WIN3B_REF_VARIANT = "flatten_base_base"

WIN3B_WIN_ID = "3B"
WIN3B_ARCHITECTURE_FACTOR = "receptive_field_spatial_resolution"
WIN3B_HYPOTHESIS = "rf_or_downsampling_limits_delta_localisation"
WIN3B_EXPERIMENT_PREFIX = "win3b"


def win3b_task_specs(**kwargs):
    """Train / val / test tasks for Milestone 8 Step 3B (mechanism protocol, ``per_image_minmax``)."""
    return mechanism_task_specs(**kwargs)


def win3b_results_dir(repo_root: Path | str) -> Path:
    return study_results_dir(repo_root, WIN3B_SUBDIR)


aggregate_win3b_runs = aggregate_runs


def load_win3b_results(
    repo_root: Path | str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    return load_study_results(
        win3b_results_dir(repo_root),
        summary_name=WIN3B_CSV,
        runs_name=WIN3B_RUNS_CSV,
    )


def run_win3b_receptive_field_study(
    train_ds,
    val_ds,
    train_task,
    *,
    device,
    num_epochs: int = 40,
    batch_size: int = 16,
    early_stop_patience: int = 25,
    configs: Sequence[SingleViewArchConfig] | None = None,
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
        configs=configs if configs is not None else win3b_receptive_field_grid(),
        win=WIN3B_WIN_ID,
        architecture_factor=WIN3B_ARCHITECTURE_FACTOR,
        hypothesis=WIN3B_HYPOTHESIS,
        experiment_prefix=WIN3B_EXPERIMENT_PREFIX,
        summary_csv_name=WIN3B_CSV,
        runs_csv_name=WIN3B_RUNS_CSV,
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        results_dir=results_dir,
        write_csv=write_csv,
        n_repeat=n_repeat,
        base_seed=base_seed,
    )


def plot_win3b_receptive_field_results(
    results_df: pd.DataFrame,
    histories: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    *,
    runs_df: pd.DataFrame | None = None,
) -> tuple[Any, Any | None]:
    val_col, val_std_col = validation_rmse_columns(results_df)
    by_var = results_df.set_index("variant")

    fig_bars, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)

    def _bar_axis(ax, names: Sequence[str], title: str) -> None:
        means = [float(by_var.loc[name, val_col]) for name in names]
        yerr = None
        if val_std_col is not None and val_std_col in by_var.columns:
            yerr = [float(by_var.loc[name, val_std_col]) for name in names]
        colors = ["C2" if "fourier" in name else "C1" for name in names]
        xs = np.arange(len(names))
        ax.bar(
            xs,
            means,
            yerr=yerr,
            capsize=4 if yerr is not None else 0,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        if runs_df is not None and not runs_df.empty and "seed" in runs_df.columns:
            run_val_col = (
                "validation_RMSE_total"
                if "validation_RMSE_total" in runs_df.columns
                else val_col
            )
            for idx, name in enumerate(names):
                pts = runs_df.loc[runs_df["variant"] == name, run_val_col].astype(float)
                if pts.empty:
                    continue
                offsets = (np.arange(len(pts)) - (len(pts) - 1) / 2.0) * 0.08
                ax.scatter(
                    np.full(len(pts), idx) + offsets,
                    pts.to_numpy(),
                    s=18,
                    color="black",
                    alpha=0.45,
                    zorder=3,
                )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [name.replace("fourier_", "").replace("flatten_", "FL_") for name in names],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel("validation RMSE total")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if WIN3B_REF_VARIANT in by_var.index:
            ref = float(by_var.loc[WIN3B_REF_VARIANT, val_col])
            ref_std = (
                float(by_var.loc[WIN3B_REF_VARIANT, val_std_col])
                if val_std_col is not None and val_std_col in by_var.columns
                else None
            )
            ax.axhline(ref, color="C1", linestyle="--", linewidth=1.2, label=f"{WIN3B_REF_VARIANT} ref")
            if ref_std is not None and ref_std > 0:
                ax.axhspan(ref - ref_std, ref + ref_std, color="C1", alpha=0.12)
            ax.legend(fontsize=8)

    _bar_axis(axes[0], WIN3B_DEPTH_ORDER, "Depth / RF (downsample=base)")
    _bar_axis(axes[1], WIN3B_DOWNSAMPLE_ORDER, "Downsampling (channels=base)")
    title_suffix = ""
    if "n_repeat" in results_df.columns:
        n_rep = int(results_df["n_repeat"].max())
        if n_rep > 1:
            title_suffix = f" (mean ± std, n={n_rep})"
    fig_bars.suptitle(
        "Milestone 8 Step 3B — receptive field / spatial resolution (Fourier + Flatten ref)"
        + title_suffix
    )
    fig_bars.tight_layout()

    fig_loss = None
    if histories:
        fig_loss, ax = plt.subplots(figsize=(8.5, 4.2))
        for name, hist in histories.items():
            if not hist:
                continue
            ys = [step["train_loss"] for step in hist]
            style = "--" if name.startswith("flatten") else "-"
            ax.plot(ys, style, label=name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("train MSE")
        ax.set_title("Milestone 8 Step 3B train loss curves (last repeat)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig_loss.tight_layout()
    return fig_bars, fig_loss


def _axis_comparison_text(
    names: Sequence[str],
    values: Mapping[str, float],
    stds: Mapping[str, float] | None,
) -> str:
    present = [name for name in names if name in values]
    if len(present) < 2:
        return ""
    best = min(present, key=lambda name: values[name])
    worst = max(present, key=lambda name: values[name])
    delta = values[worst] - values[best]
    if stds is not None:
        combined = float(
            np.sqrt(sum(stds.get(name, 0.0) ** 2 for name in (best, worst)))
        )
        if delta <= combined:
            verdict = f"Δmean={delta:.3f} within combined std≈{combined:.3f} → inconclusive"
        else:
            verdict = f"Δmean={delta:.3f} exceeds combined std≈{combined:.3f} → {best} better than {worst}"
    else:
        verdict = f"Δ={delta:.3f} ({worst} − {best})"
    return (
        f"best={best} (mean val RMSE {values[best]:.3f}); "
        f"{verdict}."
    )


def summarize_win3b_receptive_field(results_df: pd.DataFrame) -> list[str]:
    val_col, val_std_col = validation_rmse_columns(results_df)
    by_var = results_df.set_index("variant")
    bullets: list[str] = []

    if "n_repeat" in results_df.columns:
        n_rep = int(results_df["n_repeat"].max())
        if n_rep > 1:
            bullets.append(f"Aggregated over n={n_rep} training seeds (mean ± std validation RMSE).")

    def _vals(names: Sequence[str]) -> tuple[dict[str, float], dict[str, float] | None]:
        values = {
            name: float(by_var.loc[name, val_col])
            for name in names
            if name in by_var.index
        }
        stds = None
        if val_std_col is not None and val_std_col in by_var.columns:
            stds = {
                name: float(by_var.loc[name, val_std_col])
                for name in names
                if name in by_var.index
            }
        return values, stds

    depth_vals, depth_stds = _vals(WIN3B_DEPTH_ORDER)
    if depth_vals:
        bullets.append("Depth / RF (downsample=base): " + _axis_comparison_text(
            WIN3B_DEPTH_ORDER, depth_vals, depth_stds
        ))

    ds_vals, ds_stds = _vals(WIN3B_DOWNSAMPLE_ORDER)
    if ds_vals:
        bullets.append("Downsampling axis: " + _axis_comparison_text(
            WIN3B_DOWNSAMPLE_ORDER, ds_vals, ds_stds
        ))

    if WIN3B_REF_VARIANT in by_var.index:
        ref = float(by_var.loc[WIN3B_REF_VARIANT, val_col])
        ref_std = (
            float(by_var.loc[WIN3B_REF_VARIANT, val_std_col])
            if val_std_col is not None and val_std_col in by_var.columns
            else 0.0
        )
        fourier_rows = results_df[results_df["head_type"] == "fourier"]
        if not fourier_rows.empty:
            best_f = fourier_rows.sort_values(val_col).iloc[0]
            gap = float(best_f[val_col]) - ref
            gap_std = float(
                np.sqrt(
                    (best_f[val_std_col] if val_std_col else 0.0) ** 2 + ref_std ** 2
                )
            )
            if val_std_col and gap_std > 0 and abs(gap) <= gap_std:
                gap_txt = f"{gap:+.3f} vs Flatten (within combined std≈{gap_std:.3f})"
            else:
                gap_txt = f"{gap:+.3f} vs Flatten reference (mean val RMSE {ref:.3f})"
            bullets.append(f"Best Fourier ({best_f['variant']}) is {gap_txt}.")
        bullets.append(
            "Relative to Milestone 8 Step 3A: RF / resolution is a secondary check — "
            "large gaps vs Flatten usually reflect spatial readout, not RF alone."
        )
    return bullets
