"""Root-mean-square error (RMSE) and learning-rate plots for illumination-only fusion experiment notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomography_ml_validation.run_history import load_summary_for_plots

try:
    from gummybear.paths import repo_relative_path as _repo_relative_path
except ImportError:  # pragma: no cover

    def _repo_relative_path(path: Path | str) -> str:
        return str(path)


def _legend_if_labeled(ax: Any, **kwargs: Any) -> None:
    """Call ``ax.legend`` only when at least one artist has a label."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **kwargs)


@dataclass(frozen=True)
class IlluminationFusionPlotConfig:
    """Labels, variant order, and PNG filename prefix for illumination fusion plots.

    Attributes:
        title_prefix: Figure title stem prepended to plot titles.
        file_prefix: Output filename prefix under ``results_dir``.
        short_labels: ``variant_id`` → short bar-chart label.
        order_fourier: Fourier backbone variants in display order.
        order_pairs: ``(fourier_id, pooled_id, short_tag)`` tuples for
            side-by-side Fourier vs pooled comparisons.

    Examples:
        ``title_prefix="10_1A frozen"`` for the frozen fusion notebook.
    """

    title_prefix: str
    file_prefix: str
    short_labels: Mapping[str, str]
    order_fourier: Sequence[str]
    order_pairs: Sequence[tuple[str, str, str]] = field(default_factory=tuple)

    def out_path(self, results_dir: Path, stem: str) -> Path:
        """Build the PNG path for one plot stem under ``results_dir``.

        Args:
            results_dir: Notebook results directory (created by callers).
            stem: Logical plot name without extension (for example
                ``"rmse_total_barplot"``).

        Returns:
            pathlib.Path: ``{results_dir}/{file_prefix}_{stem}.png``.
        """
        return Path(results_dir) / f"{self.file_prefix}_{stem}.png"


# Shared A/B baseline ids (single-light / xyz-mean) across 10_1A and 10_1B.
_BASE_A = "m10_1a_single_illumination"
_BASE_B = "m10_1b_mean_xyz_illuminations"

# Frozen-encoder illumination fusion notebook (10_1A): Fourier A–D + pooled pairs.
PLOT_CONFIG_10_1A = IlluminationFusionPlotConfig(
    title_prefix="10_1A frozen",
    file_prefix="m10_1a",
    short_labels={
        _BASE_A: "A single L",
        _BASE_B: "B xyz mean",
        "m10_1a_c_frozen_illumination_fusion": "C frozen",
        "m10_1a_d_frozen_illumination_angle_fusion": "D frozen FiLM∠",
        f"{_BASE_A}_pooled": "A pooled",
        f"{_BASE_B}_pooled": "B pooled mean",
        "m10_1a_c_frozen_illumination_fusion_pooled": "C frozen pooled",
        "m10_1a_d_frozen_illumination_angle_fusion_pooled": "D frozen pooled FiLM∠",
    },
    order_fourier=(
        _BASE_A,
        _BASE_B,
        "m10_1a_c_frozen_illumination_fusion",
        "m10_1a_d_frozen_illumination_angle_fusion",
    ),
    order_pairs=(
        (_BASE_A, f"{_BASE_A}_pooled", "A"),
        (_BASE_B, f"{_BASE_B}_pooled", "B"),
        (
            "m10_1a_c_frozen_illumination_fusion",
            "m10_1a_c_frozen_illumination_fusion_pooled",
            "C",
        ),
        (
            "m10_1a_d_frozen_illumination_angle_fusion",
            "m10_1a_d_frozen_illumination_angle_fusion_pooled",
            "D",
        ),
    ),
)
# Plot config for notebook 10_1A (frozen encoder, Fourier A–D + pooled controls).

# End-to-end illumination fusion notebook (10_1B): learned C/D variants + pooled.
PLOT_CONFIG_10_1B = IlluminationFusionPlotConfig(
    title_prefix="10_1B e2e",
    file_prefix="m10_1b",
    short_labels={
        _BASE_A: "A single L",
        _BASE_B: "B xyz mean",
        "m10_1c_e2e_illumination_fusion": "C learned",
        "m10_1d_e2e_illumination_angle_fusion": "D +light∠",
        f"{_BASE_A}_pooled": "A pooled",
        f"{_BASE_B}_pooled": "B pooled mean",
        "m10_1c_e2e_illumination_fusion_pooled": "C pooled",
        "m10_1d_e2e_illumination_angle_fusion_pooled": "D pooled +∠",
    },
    order_fourier=(
        _BASE_A,
        _BASE_B,
        "m10_1c_e2e_illumination_fusion",
        "m10_1d_e2e_illumination_angle_fusion",
    ),
    order_pairs=(
        (_BASE_A, f"{_BASE_A}_pooled", "A"),
        (_BASE_B, f"{_BASE_B}_pooled", "B"),
        (
            "m10_1c_e2e_illumination_fusion",
            "m10_1c_e2e_illumination_fusion_pooled",
            "C",
        ),
        (
            "m10_1d_e2e_illumination_angle_fusion",
            "m10_1d_e2e_illumination_angle_fusion_pooled",
            "D",
        ),
    ),
)
# Plot config for notebook 10_1B (end-to-end fusion, learned C/D + pooled controls).


def _yerr(stds: np.ndarray, n_runs: int | None) -> np.ndarray | None:
    """Hide error bars when n<2 (std is not meaningful)."""
    stds = np.asarray(stds, dtype=float)
    if n_runs is None or int(n_runs) < 2:
        return None
    return stds


def _annotate_bars(
    ax: plt.Axes,
    x: np.ndarray,
    means: Sequence[float],
    stds: Sequence[float],
    n_runs: int | None,
) -> None:
    for xi, mu, sd in zip(x, means, stds):
        if not np.isfinite(mu):
            continue
        if n_runs is not None and int(n_runs) >= 2 and np.isfinite(sd) and sd > 0:
            label = f"{mu:.3f}±{sd:.3f}"
            y = float(mu) + float(sd)
        else:
            label = f"{mu:.3f}"
            y = float(mu)
        ax.text(xi, y, label, ha="center", va="bottom", fontsize=7)


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(("true", "1", "yes"))


def _mean_std_cols(df: pd.DataFrame, split_key: str) -> tuple[str, str]:
    col_m = (
        f"RMSE_{split_key}_mean"
        if f"RMSE_{split_key}_mean" in df.columns
        else f"RMSE_{split_key}_total_mean"
    )
    col_s = (
        f"RMSE_{split_key}_std"
        if f"RMSE_{split_key}_std" in df.columns
        else f"RMSE_{split_key}_total_std"
    )
    return col_m, col_s


def plot_stage_b_lr_study(
    *,
    results_dir: Path | str,
    config: IlluminationFusionPlotConfig,
    lr_study_df: pd.DataFrame | None = None,
    lr_study_path: Path | str | None = None,
    show: bool = True,
) -> Path | None:
    """Plot Stage B learning-rate sweeps when multiple LRs are present.

    Reads an in-memory dataframe or CSV with columns ``variant_tag``, ``lr``, and
    ``best_val_rmse``. Skips plotting when only one LR is recorded. Marks rows
    with ``used_for_eval=True`` using open circle markers.

    Args:
        results_dir: Directory for the output PNG.
        config: Plot labels and ``out_path`` prefix.
        lr_study_df: Optional pre-loaded LR study table.
        lr_study_path: Optional CSV path when dataframe not supplied.
        show: If True, call ``plt.show()``; otherwise close the figure.

    Returns:
        pathlib.Path or None: Written PNG path, or None when no sweep exists.
    """
    results_dir = Path(results_dir)
    if lr_study_df is not None and len(lr_study_df):
        lr_plot = lr_study_df.copy()
    elif lr_study_path is not None and Path(lr_study_path).is_file():
        lr_plot = pd.read_csv(lr_study_path)
        print(f"Loaded {_repo_relative_path(lr_study_path)}")
    else:
        print("No LR study CSV yet.")
        return None

    if lr_plot["lr"].nunique() <= 1:
        print("LR study has single-LR rows only (no sweep).")
        return None

    style = {
        "C_Fourier": ("C0", "C Fourier"),
        "D_Fourier": ("C1", "D Fourier"),
        "C_pooled": ("C2", "C pooled"),
        "D_pooled": ("C3", "D pooled"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    plotted = 0
    for tag, (color, label) in style.items():
        sub = lr_plot[lr_plot["variant_tag"] == tag].sort_values("lr")
        if len(sub) < 2:
            continue
        ax.plot(sub["lr"], sub["best_val_rmse"], "o-", color=color, label=label)
        plotted += 1
        if "used_for_eval" in sub.columns:
            used = sub[_as_bool_series(sub["used_for_eval"])]
            for _, row in used.iterrows():
                ax.scatter(
                    [row["lr"]],
                    [row["best_val_rmse"]],
                    s=90,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=2,
                    zorder=5,
                )
    if plotted == 0:
        plt.close(fig)
        print("LR study: no variant has ≥2 learning rates to plot.")
        return None
    ax.set_xscale("log")
    ax.set_xlabel("Stage B learning rate")
    ax.set_ylabel("best val RMSE")
    ax.set_title(f"{config.title_prefix} Stage B LR studies")
    ax.grid(True, which="both", alpha=0.3)
    _legend_if_labeled(ax, fontsize=8)
    out_lr = config.out_path(results_dir, "lr_study_barplot")
    fig.savefig(out_lr, dpi=150)
    print(f"Wrote {_repo_relative_path(out_lr)}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_lr


def plot_illumination_fusion_results(
    *,
    results_dir: Path | str,
    config: IlluminationFusionPlotConfig,
    history_path: Path | str | None = None,
    session_summary_path: Path | str | None = None,
    comparison_path: Path | str | None = None,
    lr_study_path: Path | str | None = None,
    session_run_ids: Sequence[int] | None = None,
    session_summary_df: pd.DataFrame | None = None,
    comparison_df: pd.DataFrame | None = None,
    lr_study_df: pd.DataFrame | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Draw mean±std root-mean-square error (RMSE) bar charts (and optional LR study) for 10_1 notebooks.

    Prefers aggregated run history or session summary for error bars; falls back
    to a single-run comparison CSV table (no std). Writes PNGs under ``results_dir``
    using ``config.out_path``.

    Args:
        results_dir: Notebook artifact directory.
        config: Variant order, labels, and filename prefix.
        history_path: Run history CSV table (default ``{file_prefix}_run_history.csv``).
        session_summary_path: Pre-aggregated session summary CSV table.
        comparison_path: Last-run-only comparison CSV table fallback.
        lr_study_path: Stage B LR sweep CSV table.
        session_run_ids: Optional filter to specific ``run_id`` values.
        session_summary_df, comparison_df, lr_study_df: In-memory alternatives.
        show: If True, display figures; otherwise save and close.

    Returns:
        dict: ``summary_df`` (aggregated table or None) and ``summary_src``
        (human-readable provenance string).

    Reads Stage-B result tables written via :func:`~tomography_ml_validation.run_history.append_run_history`
    and :func:`~tomography_ml_validation.run_history.build_history_row`.
    """
    results_dir = Path(results_dir)
    if history_path is None:
        history_path = results_dir / f"{config.file_prefix}_run_history.csv"
    else:
        history_path = Path(history_path)
    if session_summary_path is None:
        session_summary_path = results_dir / f"{config.file_prefix}_session_summary.csv"
    else:
        session_summary_path = Path(session_summary_path)
    if comparison_path is None:
        comparison_path = results_dir / f"{config.file_prefix}_comparison.csv"
    else:
        comparison_path = Path(comparison_path)
    if lr_study_path is None:
        lr_study_path = results_dir / f"{config.file_prefix}_lr_study.csv"
    else:
        lr_study_path = Path(lr_study_path)

    short = dict(config.short_labels)
    order_f = list(config.order_fourier)
    order_pairs = list(config.order_pairs)

    summary_df, summary_src = load_summary_for_plots(
        history_path,
        session_run_ids=session_run_ids,
        session_summary=session_summary_df,
        session_summary_path=session_summary_path,
    )
    if summary_df is None:
        print("No run history / session summary — cannot draw mean±std bars yet.")

    if summary_df is not None and len(summary_df):
        print(f"Plot summary source: {summary_src}")
        n_runs = (
            int(summary_df["n_runs"].iloc[0])
            if "n_runs" in summary_df.columns
            else 1
        )
        print(f"n_runs per variant (first)={n_runs}")

        fourier = (
            summary_df[summary_df["backbone_kind"] == "fourier"]
            if "backbone_kind" in summary_df.columns
            else summary_df
        )

        split_keys = [
            s
            for s in ("validation", "test")
            if f"RMSE_{s}_mean" in fourier.columns
            or f"RMSE_{s}_total_mean" in fourier.columns
        ]
        if split_keys:
            fig, axes = plt.subplots(
                1,
                len(split_keys),
                figsize=(5.0 * len(split_keys), 4.4),
                sharey=True,
                constrained_layout=True,
            )
            if len(split_keys) == 1:
                axes = [axes]
            for ax, split_key in zip(axes, split_keys):
                col_m, col_s = _mean_std_cols(fourier, split_key)
                sub = (
                    fourier.set_index("variant_id")
                    .reindex([v for v in order_f if v in set(fourier["variant_id"])])
                    .reset_index()
                )
                labels = [short.get(v, v) for v in sub["variant_id"]]
                means = sub[col_m].to_numpy(dtype=float)
                stds = (
                    sub[col_s].to_numpy(dtype=float)
                    if col_s in sub.columns
                    else np.zeros_like(means)
                )
                x = np.arange(len(labels))
                ax.bar(
                    x,
                    means,
                    yerr=_yerr(stds, n_runs),
                    capsize=4,
                    color=["C0", "C1", "C2", "C3"][: len(labels)],
                    edgecolor="none",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=15, ha="right")
                ax.set_ylabel("RMSE total")
                ax.set_title(f"{split_key}  mean±std  (n={n_runs})")
                ax.grid(True, axis="y", alpha=0.3)
                _annotate_bars(ax, x, means, stds, n_runs)
            fig.suptitle(
                f"{config.title_prefix} — Fourier A–D RMSE (mean ± std)",
                fontsize=12,
            )
            out = config.out_path(results_dir, "rmse_total_barplot")
            fig.savefig(out, dpi=150)
            print(f"Wrote {_repo_relative_path(out)}")
            if show:
                plt.show()
            else:
                plt.close(fig)

        axis_ok = all(
            f"RMSE_test_{ax}_mean" in fourier.columns for ax in ("X", "Y", "Z")
        )
        if axis_ok:
            sub = (
                fourier.set_index("variant_id")
                .reindex([v for v in order_f if v in set(fourier["variant_id"])])
                .reset_index()
            )
            labels = [short.get(v, v) for v in sub["variant_id"]]
            x = np.arange(len(labels))
            width = 0.22
            fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
            for k, axis in enumerate(("X", "Y", "Z")):
                means = sub[f"RMSE_test_{axis}_mean"].to_numpy(dtype=float)
                stds = sub[f"RMSE_test_{axis}_std"].to_numpy(dtype=float)
                ax.bar(
                    x + (k - 1) * width,
                    means,
                    width=width,
                    yerr=_yerr(stds, n_runs),
                    capsize=3,
                    label=axis,
                    color=f"C{k}",
                    edgecolor="none",
                )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_ylabel("RMSE")
            ax.set_title(
                f"{config.title_prefix} Fourier per-axis test RMSE "
                f"mean±std (n={n_runs})"
            )
            _legend_if_labeled(ax, fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            out2 = config.out_path(results_dir, "rmse_axis_barplot")
            fig.savefig(out2, dpi=150)
            print(f"Wrote {_repo_relative_path(out2)}")
            if show:
                plt.show()
            else:
                plt.close(fig)

        if "backbone_kind" in summary_df.columns and order_pairs:
            idx = summary_df.set_index("variant_id")
            fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
            x = np.arange(len(order_pairs))
            width = 0.35
            f_means, f_stds, p_means, p_stds, labels = [], [], [], [], []
            for vf, vp, lab in order_pairs:
                labels.append(lab)

                def _get(vid: str) -> tuple[float, float]:
                    if vid not in idx.index:
                        return float("nan"), 0.0
                    col_m, col_s = _mean_std_cols(idx, "test")
                    return float(idx.loc[vid, col_m]), float(idx.loc[vid, col_s])

                fm, fs = _get(vf)
                pm, ps = _get(vp)
                f_means.append(fm)
                f_stds.append(fs)
                p_means.append(pm)
                p_stds.append(ps)
            ax.bar(
                x - width / 2,
                f_means,
                width,
                yerr=_yerr(np.asarray(f_stds), n_runs),
                capsize=3,
                label="Fourier",
                color="C0",
                edgecolor="none",
            )
            ax.bar(
                x + width / 2,
                p_means,
                width,
                yerr=_yerr(np.asarray(p_stds), n_runs),
                capsize=3,
                label="pooled",
                color="C3",
                edgecolor="none",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("RMSE test total")
            ax.set_title(
                f"{config.title_prefix} Fourier vs pooled — "
                f"test mean±std (n={n_runs})"
            )
            _legend_if_labeled(ax, fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            _annotate_bars(ax, x - width / 2, f_means, f_stds, n_runs)
            _annotate_bars(ax, x + width / 2, p_means, p_stds, n_runs)
            out3 = config.out_path(results_dir, "fourier_vs_pooled_rmse")
            fig.savefig(out3, dpi=150)
            print(f"Wrote {_repo_relative_path(out3)}")
            if show:
                plt.show()
            else:
                plt.close(fig)

        show_cols = [
            c
            for c in (
                "variant_id",
                "backbone_kind",
                "n_runs",
                "RMSE_validation_mean",
                "RMSE_validation_std",
                "RMSE_test_mean",
                "RMSE_test_std",
            )
            if c in summary_df.columns
        ]
        print(summary_df[show_cols].to_string(index=False))
    else:
        if comparison_df is not None and len(comparison_df):
            plot_df = comparison_df.copy()
        elif comparison_path.is_file():
            plot_df = pd.read_csv(comparison_path)
            print(
                f"Loaded {_repo_relative_path(comparison_path)} "
                "(last run only, no std)"
            )
        else:
            plot_df = None
        if plot_df is not None and len(plot_df):
            if "backbone_kind" not in plot_df.columns:
                plot_df["backbone_kind"] = plot_df["variant_id"].map(
                    lambda v: "pooled"
                    if str(v).endswith("_pooled")
                    else "fourier"
                )
            plot_df["label"] = plot_df["variant_id"].map(lambda v: short.get(v, v))
            splits = [
                s for s in ("validation", "test") if s in set(plot_df["split"])
            ]
            fourier = plot_df[plot_df["backbone_kind"] == "fourier"]
            fig, axes = plt.subplots(
                1,
                max(len(splits), 1),
                figsize=(4.2 * max(len(splits), 1), 4.2),
                sharey=True,
                constrained_layout=True,
            )
            if len(splits) == 1:
                axes = [axes]
            for ax, split in zip(axes, splits):
                sub = fourier[fourier["split"] == split].drop_duplicates(
                    "variant_id", keep="last"
                )
                sub = (
                    sub.set_index("variant_id")
                    .reindex([v for v in order_f if v in set(sub["variant_id"])])
                    .reset_index()
                )
                labels = list(sub["label"])
                vals = sub["RMSE_total"].to_numpy(dtype=float)
                x = np.arange(len(labels))
                ax.bar(
                    x,
                    vals,
                    color=["C0", "C1", "C2", "C3"][: len(labels)],
                    edgecolor="none",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=15, ha="right")
                ax.set_title(f"{split} (last run only)")
                ax.grid(True, axis="y", alpha=0.3)
            fig.suptitle(
                f"{config.title_prefix} — last run only (no std yet)",
                fontsize=12,
            )
            out = config.out_path(results_dir, "rmse_total_barplot")
            fig.savefig(out, dpi=150)
            print(f"Wrote {_repo_relative_path(out)}")
            if show:
                plt.show()
            else:
                plt.close(fig)

    plot_stage_b_lr_study(
        results_dir=results_dir,
        config=config,
        lr_study_df=lr_study_df,
        lr_study_path=lr_study_path,
        show=show,
    )

    if history_path.is_file():
        hist_all = pd.read_csv(history_path)
        print(
            f"\nRun history: {_repo_relative_path(history_path)}  "
            f"rows={len(hist_all)}  run_ids="
            f"{sorted(hist_all['run_id'].unique().tolist())}"
        )

    return {"summary_df": summary_df, "summary_src": summary_src}
