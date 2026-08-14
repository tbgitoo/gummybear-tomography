"""Matplotlib helpers for M8 single-view LR and train→val/test studies."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.studies.single_view_m8 import ARCH_COLORS, ARCH_ORDER
from tomography_ml_validation.plotting.report_titles import apply_report_titles


def plot_learning_rate_study(
    lr_results: Mapping[str, Mapping[float, Mapping[str, Any]]],
    lr_by_arch: Mapping[str, float],
    *,
    lr_grid: Sequence[float],
    num_epochs: int,
    x_field: str,
    y_fields: Sequence[str],
    train_size: int,
    val_size: int,
    arch_order: Sequence[str] = ARCH_ORDER,
    arch_colors: Mapping[str, str] = ARCH_COLORS,
    title_prefix: str = "M8 LR study",
    heading: str | None = None,
    caption: str | None = None,
    figsize: tuple[float, float] = (16.0, 4.0),
) -> plt.Figure:
    """Overlay train MSE vs epoch per LR and validation MSE vs LR.

    Args:
        heading: Optional large over-title (pass from the notebook).
        caption: Optional 11px technical line under the heading. If omitted,
            built from ``title_prefix``, epoch count, fields, and split sizes.
        title_prefix: Stem of the auto-generated technical caption.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    cmap_lr = plt.cm.viridis(np.linspace(0.1, 0.9, len(lr_grid)))

    for ax, arch in zip(axes[:3], arch_order):
        for color, lr in zip(cmap_lr, lr_grid):
            curve = lr_results[arch][float(lr)]["train_curve"]
            selected = float(lr) == float(lr_by_arch[arch])
            ax.plot(
                curve,
                color=color,
                lw=2.2 if selected else 1.0,
                alpha=1.0 if selected else 0.55,
                label=f"lr={lr:g}",
            )
        ax.set_title(f"{arch} train MSE")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train MSE")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=1, loc="upper right")

    ax_sum = axes[3]
    for arch in arch_order:
        lrs = [float(lr) for lr in lr_grid]
        vals = [lr_results[arch][lr]["val_mse"] for lr in lrs]
        ax_sum.plot(
            lrs,
            vals,
            marker="o",
            color=arch_colors[arch],
            label=f"{arch} (best={lr_by_arch[arch]:g})",
        )
    ax_sum.set_xscale("log")
    ax_sum.set_yscale("log")
    ax_sum.set_xlabel("learning rate")
    ax_sum.set_ylabel("validation MSE")
    ax_sum.set_title("Validation MSE vs LR")
    ax_sum.grid(True, which="both", alpha=0.3)
    ax_sum.legend(fontsize=8)

    technical = caption or (
        f"{title_prefix}  |  {num_epochs} epochs  |  "
        f"x={x_field}  y={list(y_fields)}  |  "
        f"train n={train_size}  val n={val_size}"
    )
    if heading:
        apply_report_titles(
            fig, heading, technical, steal_existing=False
        )
    else:
        fig.suptitle(technical, fontsize=11)
        fig.tight_layout()
    return fig


def plot_rmse_summary_bars(
    summary_df: pd.DataFrame,
    *,
    variant_prefix: str,
    arch_order: Sequence[str] = ARCH_ORDER,
    arch_colors: Mapping[str, str] = ARCH_COLORS,
    title: str | None = None,
    figsize: tuple[float, float] = (7.2, 4.2),
) -> tuple[plt.Figure, dict[str, Any]]:
    """Bar chart of validation/test RMSE mean ± SD across training runs."""
    variant_of = {a: f"{variant_prefix}_{a}" for a in arch_order}
    by_variant = {str(r["variant_id"]): r for _, r in summary_df.iterrows()}

    def _split_stats(split_key: str):
        means, stds, n_runs_list = [], [], []
        col_mean = f"RMSE_{split_key}_mean"
        col_std = f"RMSE_{split_key}_std"
        for arch in arch_order:
            row = by_variant[variant_of[arch]]
            means.append(float(row[col_mean]))
            stds.append(float(row[col_std]))
            n_runs_list.append(int(row["n_runs"]))
        return means, stds, n_runs_list

    val_means, val_stds, n_runs_list = _split_stats("validation")
    test_means, test_stds, _ = _split_stats("test")
    n_runs = int(n_runs_list[0])
    show_err = n_runs >= 2
    xs = np.arange(len(arch_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        xs - width / 2,
        val_means,
        width,
        yerr=np.asarray(val_stds) if show_err else None,
        color=[arch_colors[a] for a in arch_order],
        edgecolor="black",
        linewidth=0.6,
        capsize=4 if show_err else 0,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
        label="validation",
        alpha=0.95,
    )
    ax.bar(
        xs + width / 2,
        test_means,
        width,
        yerr=np.asarray(test_stds) if show_err else None,
        color=[arch_colors[a] for a in arch_order],
        edgecolor="black",
        linewidth=0.6,
        capsize=4 if show_err else 0,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
        label="test",
        alpha=0.55,
        hatch="//",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(list(arch_order))
    ax.set_ylabel("RMSE")
    ax.set_title(
        title
        or f"Validation / test RMSE by architecture (mean ± SD across runs; n={n_runs})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    for x, m, s in zip(xs - width / 2, val_means, val_stds):
        y_text = m + (s if show_err and np.isfinite(s) else 0.0)
        ax.text(x, y_text, f"{m:.3f}", ha="center", va="bottom", fontsize=7)
    for x, m, s in zip(xs + width / 2, test_means, test_stds):
        y_text = m + (s if show_err and np.isfinite(s) else 0.0)
        ax.text(x, y_text, f"{m:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    stats = {
        "val_means": val_means,
        "val_stds": val_stds,
        "test_means": test_means,
        "test_stds": test_stds,
        "n_runs_list": n_runs_list,
        "n_runs": n_runs,
    }
    return fig, stats


def plot_error_histograms(
    full_results: Mapping[str, Mapping[str, Any]],
    *,
    arch_order: Sequence[str] = ARCH_ORDER,
    arch_colors: Mapping[str, str] = ARCH_COLORS,
    title: str | None = None,
    figsize: tuple[float, float] = (10.5, 4.0),
) -> plt.Figure:
    """Validation and test per-sample L2 error histograms (last training run)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    all_val = np.concatenate([full_results[a]["val_errs"] for a in arch_order])
    all_test = np.concatenate([full_results[a]["test_errs"] for a in arch_order])
    err_lo = float(min(all_val.min(), all_test.min()))
    err_hi = float(max(all_val.max(), all_test.max()) + 1e-9)
    bins = np.linspace(err_lo, err_hi, 20)

    axes[0].hist(
        [full_results[a]["val_errs"] for a in arch_order],
        bins=bins,
        color=[arch_colors[a] for a in arch_order],
        label=[
            (
                f"{a}  mean={full_results[a]['val_errs'].mean():.3f}  "
                f"RMSE={full_results[a]['val_rmse']:.3f}"
            )
            for a in arch_order
        ],
        histtype="bar",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].set_xlabel(r"validation per-sample $\|pred - target\|_2$")
    axes[0].set_ylabel("count")
    axes[0].set_title("Validation error histogram (last run)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].hist(
        [full_results[a]["test_errs"] for a in arch_order],
        bins=bins,
        color=[arch_colors[a] for a in arch_order],
        label=[
            (
                f"{a}  mean={full_results[a]['test_errs'].mean():.3f}  "
                f"RMSE={full_results[a]['test_rmse']:.3f}"
            )
            for a in arch_order
        ],
        histtype="bar",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].set_xlabel(r"test per-sample $\|pred - target\|_2$")
    axes[1].set_ylabel("count")
    axes[1].set_title("Test error histogram (last run)")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
