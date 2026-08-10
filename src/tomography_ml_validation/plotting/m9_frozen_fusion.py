"""Matplotlib helpers for M9 frozen Fourier / pooled fusion ladders."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.localization.localize_multiview import (
    PACKING_MEAN_POOL,
    PACKING_ORDERED_CONCAT,
)

FOURIER_LR_STYLE: dict[str, tuple[str, str]] = {
    PACKING_MEAN_POOL: ("C1", "mean-pool MLP"),
    PACKING_ORDERED_CONCAT: ("C0", "ordered concat"),
    "deepsets_fourier": ("C2", "DeepSets Fourier"),
}
POOLED_LR_STYLE: dict[str, tuple[str, str]] = {
    PACKING_MEAN_POOL: ("C1", "mean-pool MLP pooled"),
    PACKING_ORDERED_CONCAT: ("C0", "ordered concat pooled"),
    "deepsets_no_fourier": ("C3", "DeepSets no-Fourier"),
}

FOURIER_LADDER_ORDER: tuple[str, ...] = (
    "SV ref",
    "xyz mean",
    "mean-pool MLP",
    "DeepSets Fourier",
    "ordered concat",
)
POOLED_LADDER_ORDER: tuple[str, ...] = (
    "SV pooled",
    "xyz mean pooled",
    "mean-pool MLP pooled",
    "DeepSets no-Fourier",
    "ordered concat pooled",
)

PARAM_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("SV ref", "SV pooled", "SV"),
    ("xyz mean", "xyz mean pooled", "xyz mean"),
    ("mean-pool MLP", "mean-pool MLP pooled", "mean-pool MLP"),
    ("DeepSets Fourier", "DeepSets no-Fourier", "DeepSets"),
    ("ordered concat", "ordered concat pooled", "ordered concat"),
)
RMSE_PAIRS: tuple[tuple[str, str], ...] = (
    ("SV ref", "SV pooled"),
    ("xyz mean", "xyz mean pooled"),
    ("mean-pool MLP", "mean-pool MLP pooled"),
    ("DeepSets Fourier", "DeepSets no-Fourier"),
    ("ordered concat", "ordered concat pooled"),
)


def ensure_display_label(df: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    """Copy ``df`` and fill ``display_label`` from ``variant_id`` when missing."""
    out = df.copy()
    if "display_label" not in out.columns:
        out["display_label"] = out["variant_id"].map(lambda v: mapping.get(v, v))
    return out


def plot_m9_lr_study(
    lr_study_df: pd.DataFrame,
    *,
    family: str,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.4),
) -> plt.Figure | None:
    """Log-x Stage-B LR vs best validation RMSE (one curve per packing)."""
    if lr_study_df is None or len(lr_study_df) == 0:
        return None
    style = FOURIER_LR_STYLE if family == "fourier" else POOLED_LR_STYLE
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plotted = False
    for packing, (color, label) in style.items():
        sub = lr_study_df[lr_study_df["packing"] == packing].sort_values("lr")
        if not len(sub):
            continue
        ax.plot(sub["lr"], sub["best_val_rmse"], "o-", color=color, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_xlabel("Stage B learning rate")
    ax.set_ylabel("best val RMSE")
    ax.set_title(title or f"09_1 Stage B LR studies ({family})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def plot_m9_rmse_ladder(
    comparison_df: pd.DataFrame,
    *,
    family: str,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Side-by-side validation/test RMSE bars for one fusion ladder."""
    order = FOURIER_LADDER_ORDER if family == "fourier" else POOLED_LADDER_ORDER
    splits = [s for s in ("validation", "test") if s in set(comparison_df["split"])]
    if not splits:
        raise ValueError("comparison_df has no validation/test rows")
    if figsize is None:
        figsize = (5.2 * max(len(splits), 1), 4.4)
    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
    )
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        sub = comparison_df[comparison_df["split"] == split].drop_duplicates(
            "display_label", keep="last"
        )
        sub = (
            sub.set_index("display_label")
            .reindex([o for o in order if o in set(sub["display_label"])])
            .reset_index()
        )
        x = np.arange(len(sub))
        bars = ax.bar(
            x,
            sub["RMSE_total"],
            color=[f"C{i}" for i in range(len(sub))],
            edgecolor="none",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(list(sub["display_label"]), rotation=20, ha="right")
        ax.set_ylabel("RMSE total")
        ax.set_title(split)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, sub["RMSE_total"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.suptitle(title or f"09_1 {family} fusion — total RMSE", fontsize=12)
    return fig


def plot_m9_param_counts_fourier_vs_pooled(
    comparison_fourier: pd.DataFrame,
    comparison_pooled: pd.DataFrame,
    *,
    title: str = "09_1 trainable params — Fourier vs non-Fourier",
    figsize: tuple[float, float] = (7.2, 4.0),
) -> plt.Figure | None:
    """Grouped log-y parameter bars (Fourier vs pooled matched heads)."""
    a = comparison_fourier[comparison_fourier["split"] == "test"].drop_duplicates(
        "display_label", keep="last"
    )
    b = comparison_pooled[comparison_pooled["split"] == "test"].drop_duplicates(
        "display_label", keep="last"
    )
    if "learned_parameter_count" not in a.columns or "learned_parameter_count" not in b.columns:
        return None
    a_map = dict(zip(a["display_label"], a["learned_parameter_count"]))
    b_map = dict(zip(b["display_label"], b["learned_parameter_count"]))
    labels, va, vb = [], [], []
    for la, lb, short in PARAM_PAIRS:
        if la in a_map and lb in b_map:
            labels.append(short)
            va.append(float(a_map[la]))
            vb.append(float(b_map[lb]))
    if not labels:
        return None
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.bar(x - w / 2, va, w, label="Fourier (09_1A)", color="C0")
    ax.bar(x + w / 2, vb, w, label="pooled / non-Fourier (09_1B)", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("trainable params")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for xi, yf, yp in zip(x, va, vb):
        ax.text(xi - w / 2, yf, f"{int(yf):,}", ha="center", va="bottom", fontsize=6)
        ax.text(xi + w / 2, yp, f"{int(yp):,}", ha="center", va="bottom", fontsize=6)
    return fig


def plot_m9_rmse_fourier_vs_pooled(
    comparison_fourier: pd.DataFrame,
    comparison_pooled: pd.DataFrame,
    *,
    split: str,
    title: str | None = None,
    figsize: tuple[float, float] = (7.2, 4.4),
) -> plt.Figure | None:
    """Grouped Fourier vs pooled RMSE bars for one split."""
    a = comparison_fourier[comparison_fourier["split"] == split].drop_duplicates(
        "display_label", keep="last"
    )
    b = comparison_pooled[comparison_pooled["split"] == split].drop_duplicates(
        "display_label", keep="last"
    )
    a_map = dict(zip(a["display_label"], a["RMSE_total"]))
    b_map = dict(zip(b["display_label"], b["RMSE_total"]))
    labels, va, vb = [], [], []
    for la, lb in RMSE_PAIRS:
        if la in a_map and lb in b_map:
            labels.append(la.replace(" Fourier", "").replace(" ref", ""))
            va.append(float(a_map[la]))
            vb.append(float(b_map[lb]))
    if not labels:
        return None
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.bar(x - w / 2, va, w, label="Fourier (09_1A)", color="C0")
    ax.bar(x + w / 2, vb, w, label="pooled (09_1B)", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("RMSE total")
    ax.set_title(title or f"09_1 Fourier vs pooled — {split}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def combine_m9_comparisons(
    comparison_fourier: pd.DataFrame,
    comparison_pooled: pd.DataFrame,
) -> pd.DataFrame:
    """Stack Fourier and pooled comparison tables with a ``family`` column."""
    a = comparison_fourier.copy()
    b = comparison_pooled.copy()
    a["family"] = "fourier"
    b["family"] = "pooled"
    return pd.concat([a, b], ignore_index=True, sort=False)
