"""Matplotlib helpers for the M9 09_0 expert-mean diagnostics."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

AXIS_LABELS = ("X", "Y", "Z")
AXIS_COLORS = ("C0", "C1", "C2")


def plot_m9_0_per_angle_experts(
    *,
    bank,
    dataset,
    x_field: str,
    angles: Sequence[float],
    device,
    n_samples: int = 3,
    y_fields: Sequence[str] = ("particle_x", "particle_y", "particle_z"),
) -> list[plt.Figure]:
    """One figure per sample: expert xyz vs angle with true and mean lines."""
    figures: list[plt.Figure] = []
    n_show = min(int(n_samples), len(dataset))
    bank.eval()
    with torch.no_grad():
        for sample_i in range(n_show):
            images, targets = dataset[sample_i]
            views = torch.as_tensor(images[x_field], dtype=torch.float32, device=device)
            if views.ndim == 4:
                views = views.unsqueeze(0)
            xyz_true = np.asarray(
                [float(targets[name]) for name in y_fields], dtype=np.float64
            )
            xyz_per_view = (
                bank.predict_per_view(views, list(angles)).squeeze(0).detach().cpu().numpy()
            )
            xyz_mean = xyz_per_view.mean(axis=0)
            seq_id = getattr(dataset.rows[sample_i], "sequence_id", sample_i)
            fig, axes = plt.subplots(
                3, 1, figsize=(9.0, 6.5), sharex=True, constrained_layout=True
            )
            for ax, axis_name, color, k in zip(axes, AXIS_LABELS, AXIS_COLORS, range(3)):
                ax.plot(
                    list(angles),
                    xyz_per_view[:, k],
                    "o-",
                    color=color,
                    markersize=4,
                    linewidth=1.2,
                    label=f"expert {axis_name}",
                )
                ax.axhline(
                    float(xyz_true[k]),
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"true {axis_name}={xyz_true[k]:.3f}",
                )
                ax.axhline(
                    float(xyz_mean[k]),
                    color=color,
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.85,
                    label=f"09_0 mean={xyz_mean[k]:.3f}",
                )
                ax.set_ylabel(axis_name)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)
            axes[-1].set_xlabel("acquisition angle (deg)")
            fig.suptitle(
                f"09_0 per-angle experts — {seq_id}  "
                f"true=[{xyz_true[0]:.3f}, {xyz_true[1]:.3f}, {xyz_true[2]:.3f}]",
                fontsize=11,
            )
            figures.append(fig)
    return figures


def collect_m9_0_bias_vs_std(
    *,
    bank,
    dataset,
    x_field: str,
    angles: Sequence[float],
    device,
    y_fields: Sequence[str] = ("particle_x", "particle_y", "particle_z"),
) -> pd.DataFrame:
    """Per-sample, per-axis bias of mean(xyz) vs expert std."""
    rows: list[dict] = []
    bank.eval()
    with torch.no_grad():
        for sample_i in range(len(dataset)):
            images, targets = dataset[sample_i]
            views = torch.as_tensor(images[x_field], dtype=torch.float32, device=device)
            if views.ndim == 4:
                views = views.unsqueeze(0)
            xyz_true = np.asarray(
                [float(targets[name]) for name in y_fields], dtype=np.float64
            )
            xyz_per_view = (
                bank.predict_per_view(views, list(angles)).squeeze(0).detach().cpu().numpy()
            )
            xyz_mean = xyz_per_view.mean(axis=0)
            xyz_std = xyz_per_view.std(axis=0, ddof=0)
            seq_id = getattr(dataset.rows[sample_i], "sequence_id", sample_i)
            for axis_name, k in zip(AXIS_LABELS, range(3)):
                rows.append(
                    {
                        "sequence_id": seq_id,
                        "axis": axis_name,
                        "true": float(xyz_true[k]),
                        "f0_mean": float(xyz_mean[k]),
                        "bias": float(xyz_mean[k] - xyz_true[k]),
                        "expert_std": float(xyz_std[k]),
                        "abs_bias": float(abs(xyz_mean[k] - xyz_true[k])),
                    }
                )
    return pd.DataFrame(rows)


def plot_m9_0_bias_vs_expert_std(bias_std_df: pd.DataFrame) -> tuple[plt.Figure, plt.Figure]:
    """Scatter bias vs expert disagreement (faceted + combined)."""
    fig_axes, axes = plt.subplots(
        1, 3, figsize=(12.0, 4.0), sharey=False, constrained_layout=True
    )
    for ax, axis_name, color in zip(axes, AXIS_LABELS, AXIS_COLORS):
        sub = bias_std_df[bias_std_df["axis"] == axis_name]
        ax.scatter(sub["expert_std"], sub["bias"], s=28, alpha=0.75, color=color, edgecolors="none")
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("expert std (across angles)")
        ax.set_ylabel("09_0 bias (mean − true)")
        ax.set_title(f"axis {axis_name}  (n={len(sub)})")
        ax.grid(True, alpha=0.3)
    fig_axes.suptitle("09_0 validation: bias of mean(xyz) vs expert disagreement", fontsize=12)

    fig_all, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    for axis_name, color in zip(AXIS_LABELS, AXIS_COLORS):
        sub = bias_std_df[bias_std_df["axis"] == axis_name]
        ax.scatter(
            sub["expert_std"],
            sub["bias"],
            s=28,
            alpha=0.7,
            color=color,
            label=axis_name,
            edgecolors="none",
        )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("expert std (across angles)")
    ax.set_ylabel("09_0 bias (mean − true)")
    ax.set_title("09_0 validation: bias vs expert std (all axes)")
    ax.legend(title="axis")
    ax.grid(True, alpha=0.3)
    return fig_axes, fig_all
