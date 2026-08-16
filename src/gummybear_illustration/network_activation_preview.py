"""Notebook inspection grids for M8 network activations (same bundle as POV)."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from .network_activations import NetworkActivationBundle


def _channel_grid(vol: np.ndarray, *, title: str, ncols: int = 8) -> None:
    arr = np.asarray(vol, dtype=float)
    n_ch = int(arr.shape[0])
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.35, nrows * 1.45))
    axes_f = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes_f):
        ax.set_axis_off()
        if i >= n_ch:
            continue
        plane = arr[i]
        ax.imshow(plane, cmap="turbo", origin="upper")
        ax.set_title(str(i), fontsize=8)
    fig.suptitle(f"{title}  {tuple(arr.shape)}  (colormap per image)", fontsize=11)
    fig.tight_layout()
    plt.show()


def _vector_strip(vec: np.ndarray, *, title: str, row_height: int = 48) -> None:
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.size > 4096:
        n_ch = 64 if v.size % 64 == 0 else int(np.round(np.sqrt(v.size)))
        if v.size % n_ch == 0:
            img = v.reshape(n_ch, -1)
        else:
            side = int(np.ceil(np.sqrt(v.size)))
            pad = np.full(side * side, np.nan, dtype=float)
            pad[: v.size] = v
            img = pad.reshape(side, side)
        fig, ax = plt.subplots(figsize=(10.0, 4.2))
        im = ax.imshow(img, cmap="turbo", aspect="auto", interpolation="nearest")
        ax.set_ylabel("channel")
        ax.set_xlabel(f"flattened H·W  (D={v.size})")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        plt.show()
        return
    img = np.tile(v[np.newaxis, :], (int(row_height), 1))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.11 * v.size), 1.6))
    im = ax.imshow(img, cmap="turbo", aspect="auto", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel(f"unit index  (D={v.size})")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    plt.show()


def _xyz_compare(pred: np.ndarray, true: np.ndarray | None, *, title: str) -> None:
    pred = np.asarray(pred, dtype=float).reshape(3)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    idx = np.arange(3)
    width = 0.35
    ax.bar(idx - width / 2, pred, width, label="predicted", color="#2e8b57")
    if true is not None:
        t = np.asarray(true, dtype=float).reshape(3)
        ax.bar(idx + width / 2, t, width, label="target", color="#888888")
        print(title)
        print("  predicted xyz:", pred)
        print("  target xyz:   ", t)
        print("  error xyz:    ", pred - t)
    else:
        print(title, "predicted xyz:", pred)
    ax.set_xticks(idx, ["x", "y", "z"])
    ax.set_ylabel("mm")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()


def _readout_blocks(
    *,
    heading: str,
    conv_maps: tuple[np.ndarray, ...],
    prepool: np.ndarray,
    prepool_title: str,
    pooled: np.ndarray,
    pooled_title: str,
    mlp_stages: tuple[tuple[str, np.ndarray], ...],
    y_pred: np.ndarray,
    y_true: np.ndarray | None,
    xyz_title: str,
) -> None:
    print(heading)
    c16, c32, c64 = conv_maps
    _channel_grid(c16, title=f"{heading}: CNN block 1 (16 × 128 × 128)")
    _channel_grid(c32, title=f"{heading}: CNN block 2 (32 × 128 × 128)")
    _channel_grid(c64, title=f"{heading}: CNN block 3 (64 × 128 × 128)")
    _channel_grid(prepool, title=prepool_title)
    _vector_strip(pooled, title=pooled_title, row_height=64)
    for name, vec in mlp_stages:
        _vector_strip(vec, title=f"{heading}: {name}", row_height=max(40, int(np.sqrt(vec.size) * 8)))
    _xyz_compare(y_pred, y_true, title=xyz_title)


def show_bundle_inspections(bundle: NetworkActivationBundle) -> None:
    """Display CNN / Fourier / GAP / MLP / xyz panels for one illustration sample."""
    print(f"activation source: {bundle.source}")
    _readout_blocks(
        heading="Fourier",
        conv_maps=bundle.conv_maps,
        prepool=bundle.fourier_prepool,
        prepool_title="Fourier term × activation (before spatial mean)",
        pooled=bundle.fourier_pooled,
        pooled_title="Fourier pooled embedding  [64]",
        mlp_stages=bundle.mlp_stages,
        y_pred=bundle.y_pred,
        y_true=bundle.y_true,
        xyz_title="Fourier: predicted vs target particle coordinates",
    )
    _readout_blocks(
        heading="Average pooling (GAP)",
        conv_maps=bundle.gap_conv_maps,
        prepool=bundle.gap_prepool,
        prepool_title="GAP term × activation (constant-1 basis, before spatial mean)",
        pooled=bundle.gap,
        pooled_title="GAP pooled embedding  [64]",
        mlp_stages=bundle.gap_mlp_stages,
        y_pred=bundle.y_pred_pooled,
        y_true=bundle.y_true,
        xyz_title="GAP: predicted vs target particle coordinates",
    )
    _readout_blocks(
        heading="Flatten",
        conv_maps=bundle.flatten_conv_maps,
        prepool=bundle.flatten_pre,
        prepool_title="Flatten spatial maps (before Flatten(1))",
        pooled=bundle.flatten_vec,
        pooled_title="Flatten readout  [C·H·W]",
        mlp_stages=bundle.flatten_mlp_stages,
        y_pred=bundle.y_pred_flatten,
        y_true=bundle.y_true,
        xyz_title="Flatten: predicted vs target particle coordinates",
    )
