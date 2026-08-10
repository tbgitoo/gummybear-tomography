"""Matplotlib helpers for M9 e2e geometry-aware Fourier / pooled ladders."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml.studies.m9_e2e_geometry_fusion import (
    FOURIER_DISPLAY,
    POOLED_DISPLAY,
)
from tomography_ml_validation.plotting.m9_frozen_fusion import ensure_display_label

FOURIER_E2E_LADDER_ORDER: tuple[str, ...] = (
    "SV ref",
    "xyz mean",
    "09_2 compact",
    "09_3 large",
)
POOLED_E2E_LADDER_ORDER: tuple[str, ...] = (
    "SV pooled",
    "xyz mean pooled",
    "09_2 compact pooled",
    "09_3 large pooled",
)

E2E_PARAM_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("SV ref", "SV pooled", "SV"),
    ("xyz mean", "xyz mean pooled", "xyz mean"),
    ("09_2 compact", "09_2 compact pooled", "09_2 compact"),
    ("09_3 large", "09_3 large pooled", "09_3 large"),
)
E2E_RMSE_PAIRS: tuple[tuple[str, str, str], ...] = E2E_PARAM_PAIRS


def ensure_e2e_display_label(df: pd.DataFrame, *, family: str) -> pd.DataFrame:
    """Fill ``display_label`` for an e2e comparison table."""
    mapping: Mapping[str, str] = (
        FOURIER_DISPLAY if family == "fourier" else POOLED_DISPLAY
    )
    return ensure_display_label(df, mapping)


def plot_m9_e2e_rmse_ladder(
    comparison_df: pd.DataFrame,
    *,
    family: str,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Side-by-side validation/test RMSE bars for one e2e geometry ladder."""
    order = (
        FOURIER_E2E_LADDER_ORDER if family == "fourier" else POOLED_E2E_LADDER_ORDER
    )
    plot_df = ensure_e2e_display_label(comparison_df, family=family)
    splits = [s for s in ("validation", "test") if s in set(plot_df["split"])]
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
        sub = plot_df[plot_df["split"] == split].drop_duplicates(
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
    fig.suptitle(
        title or f"09_2 {family} e2e + geometry — total RMSE", fontsize=12
    )
    return fig


def plot_m9_e2e_param_counts_fourier_vs_pooled(
    comparison_fourier: pd.DataFrame,
    comparison_pooled: pd.DataFrame,
    *,
    title: str = "09_2/09_3 trainable params — Fourier vs pooled",
    figsize: tuple[float, float] = (7.2, 4.0),
) -> plt.Figure | None:
    """Grouped log-y parameter bars for matched e2e heads."""
    a = ensure_e2e_display_label(comparison_fourier, family="fourier")
    b = ensure_e2e_display_label(comparison_pooled, family="pooled")
    a = a[a["split"] == "test"].drop_duplicates("display_label", keep="last")
    b = b[b["split"] == "test"].drop_duplicates("display_label", keep="last")
    if "learned_parameter_count" not in a.columns or "learned_parameter_count" not in b.columns:
        return None
    a_map = dict(zip(a["display_label"], a["learned_parameter_count"]))
    b_map = dict(zip(b["display_label"], b["learned_parameter_count"]))
    labels, va, vb = [], [], []
    for la, lb, short in E2E_PARAM_PAIRS:
        if la in a_map and lb in b_map:
            labels.append(short)
            va.append(float(a_map[la]))
            vb.append(float(b_map[lb]))
    if not labels:
        return None
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.bar(x - w / 2, va, w, label="Fourier (09_2A)", color="C0")
    ax.bar(x + w / 2, vb, w, label="pooled (09_2B)", color="C3")
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


def plot_m9_e2e_rmse_fourier_vs_pooled(
    comparison_fourier: pd.DataFrame,
    comparison_pooled: pd.DataFrame,
    *,
    split: str,
    title: str | None = None,
    figsize: tuple[float, float] = (7.2, 4.4),
) -> plt.Figure | None:
    """Grouped Fourier vs pooled RMSE bars for one split (e2e geometry)."""
    a = ensure_e2e_display_label(comparison_fourier, family="fourier")
    b = ensure_e2e_display_label(comparison_pooled, family="pooled")
    a = a[a["split"] == split].drop_duplicates("display_label", keep="last")
    b = b[b["split"] == split].drop_duplicates("display_label", keep="last")
    a_map = dict(zip(a["display_label"], a["RMSE_total"]))
    b_map = dict(zip(b["display_label"], b["RMSE_total"]))
    labels, va, vb = [], [], []
    for la, lb, short in E2E_RMSE_PAIRS:
        if la in a_map and lb in b_map:
            labels.append(short)
            va.append(float(a_map[la]))
            vb.append(float(b_map[lb]))
    if not labels:
        return None
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.bar(x - w / 2, va, w, label="Fourier (09_2A)", color="C0")
    ax.bar(x + w / 2, vb, w, label="pooled (09_2B)", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("RMSE total")
    ax.set_title(title or f"09_2/09_3 Fourier vs pooled — {split}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return fig
