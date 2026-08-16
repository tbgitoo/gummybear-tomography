"""PNG textures for the network illustration (greyscale plates + colormaps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

from .anomaly_zscore import DEFAULT_ZSCORE_CLIP, zscore_to_gray_uint8


def write_zscore_gray_png(
    z: np.ndarray,
    dest: str | Path,
    *,
    clip: float = DEFAULT_ZSCORE_CLIP,
) -> str:
    path = Path(dest)
    rgb = zscore_to_gray_uint8(z, clip=clip)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")
    return path.name


def _slice_zscore(values: np.ndarray) -> np.ndarray:
    """``(x - mean) / std`` on one slice; constant slices become zeros."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    mean = float(np.mean(arr[finite]))
    std = float(np.std(arr[finite]))
    if std <= 1e-8:
        return np.zeros_like(arr, dtype=float)
    z = (arr - mean) / std
    return np.where(finite, z, 0.0)


def _to_uint8_rgb(
    values: np.ndarray,
    *,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    scale: str = "minmax",
    zscore_clip: float = 2.0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    mode = str(scale).strip().lower()
    if mode == "zscore":
        arr = _slice_zscore(arr)
        finite = np.isfinite(arr)
        lo, hi = -float(zscore_clip), float(zscore_clip)
        if hi - lo < 1e-12:
            raise ValueError(f"zscore_clip must be > 0, got {zscore_clip}")
        t = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        t = np.where(finite, t, 0.5)
    else:
        if vmin is None or vmax is None:
            if not finite.any():
                lo, hi = 0.0, 1.0
            else:
                lo = float(np.min(arr[finite]))
                hi = float(np.max(arr[finite]))
        else:
            lo, hi = float(vmin), float(vmax)
        if hi - lo < 1e-12:
            t = np.zeros_like(arr, dtype=float)
        else:
            t = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        t = np.where(finite, t, 0.5)
    rgba = colormaps[cmap](t)
    return (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)


def write_vector_strip_png(
    vec: np.ndarray,
    dest: str | Path,
    *,
    cmap: str = "turbo",
    row_height: int = 16,
) -> str:
    """1-D activation strip; columns are units (same scale as inspection imshow)."""
    v = np.asarray(vec, dtype=float).reshape(-1)
    img = np.tile(v[np.newaxis, :], (int(row_height), 1))
    return write_colormap_png(img, dest, cmap=cmap)


def write_fourier_pane_png(
    prepool: np.ndarray,
    dest: str | Path,
    *,
    channels: tuple[int, ...] = (0, 9, 14, 61),
    cmap: str = "turbo",
    scale: str = "minmax",
    zscore_clip: float = 2.0,
) -> str:
    """2×2 channel tiles. Each tile is scaled on its own (minmax or per-slice z).

    Layout: top-left 9, top-right 0, bottom-left 14, bottom-right 61.
    """
    arr = np.asarray(prepool, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {arr.shape}")
    tiles = [
        _to_uint8_rgb(
            arr[int(ch) % arr.shape[0]],
            cmap=cmap,
            scale=scale,
            zscore_clip=zscore_clip,
        )
        for ch in channels
    ]
    if len(tiles) != 4:
        raise ValueError("Fourier pane expects four channels")
    h, w = tiles[0].shape[:2]
    gap = 10
    canvas_h = 2 * h + 3 * gap
    canvas_w = 2 * w + 3 * gap
    canvas = np.full((canvas_h, canvas_w, 3), 28, dtype=np.uint8)
    # PNG row 0 is the top of the plate. Swap the top pair vs channel order
    # (0, 9, 14, 61) so top-left is channel 9 and top-right is channel 0.
    coords = (
        (gap, 2 * gap + w),
        (gap, gap),
        (2 * gap + h, gap),
        (2 * gap + h, 2 * gap + w),
    )
    for (row, col), tile in zip(coords, tiles, strict=True):
        canvas[row : row + h, col : col + w] = tile
    path = Path(dest)
    Image.fromarray(np.ascontiguousarray(canvas), mode="RGB").save(
        path, format="PNG"
    )
    return path.name


def write_colormap_png(
    values: np.ndarray,
    dest: str | Path,
    *,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    scale: str = "minmax",
    zscore_clip: float = 2.0,
) -> str:
    """Write ``values`` with the same row order as the grayscale input plate."""
    path = Path(dest)
    rgb = _to_uint8_rgb(
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        scale=scale,
        zscore_clip=zscore_clip,
    )
    Image.fromarray(np.ascontiguousarray(rgb), mode="RGB").save(
        path, format="PNG"
    )
    return path.name


def volume_colormap_limits(vol: np.ndarray) -> tuple[float, float]:
    """Shared 2–98 percentile scale for every face of one activation tensor."""
    arr = np.asarray(vol, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    if hi - lo < 1e-12:
        return lo - 1.0, hi + 1.0
    return lo, hi


def channel_colormap_limits(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel min/max, matching inspection ``imshow(..., cmap='turbo')``."""
    arr = np.asarray(vol, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {arr.shape}")
    n_ch = int(arr.shape[0])
    vmin = np.empty(n_ch, dtype=float)
    vmax = np.empty(n_ch, dtype=float)
    for i in range(n_ch):
        plane = arr[i]
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            vmin[i], vmax[i] = 0.0, 1.0
        else:
            vmin[i] = float(np.min(finite))
            vmax[i] = float(np.max(finite))
    return vmin, vmax


def write_face_png_per_channel(
    face: np.ndarray,
    dest: str | Path,
    *,
    channel_axis: int | None,
    vmin: np.ndarray | float,
    vmax: np.ndarray | float,
    cmap: str = "turbo",
    scale: str = "minmax",
    zscore_clip: float = 2.0,
) -> str:
    """Colour a 2-D face; ``channel_axis`` strips use independent scales."""
    arr = np.asarray(face, dtype=float)
    if channel_axis is None:
        return write_colormap_png(
            arr,
            dest,
            cmap=cmap,
            vmin=float(np.asarray(vmin).reshape(-1)[0]),
            vmax=float(np.asarray(vmax).reshape(-1)[0]),
            scale=scale,
            zscore_clip=zscore_clip,
        )
    axis = int(channel_axis)
    n_ch = int(arr.shape[axis])
    lo = np.asarray(vmin, dtype=float).reshape(-1)
    hi = np.asarray(vmax, dtype=float).reshape(-1)
    if lo.size != n_ch or hi.size != n_ch:
        raise ValueError(f"expected {n_ch} channel limits, got {lo.size}/{hi.size}")
    rgb = np.empty(arr.shape + (3,), dtype=np.uint8)
    for i in range(n_ch):
        sl: list[slice | int] = [slice(None), slice(None)]
        sl[axis] = i
        key = tuple(sl)
        rgb[key] = _to_uint8_rgb(
            arr[key],
            cmap=cmap,
            vmin=float(lo[i]),
            vmax=float(hi[i]),
            scale=scale,
            zscore_clip=zscore_clip,
        )
    path = Path(dest)
    Image.fromarray(np.ascontiguousarray(rgb), mode="RGB").save(
        path, format="PNG"
    )
    return path.name


def volume_boundary_faces(vol: np.ndarray) -> dict[str, np.ndarray]:
    """Six boundary slices of ``[C, H, W]`` (actual nodes, not reductions).

    Cuboid axes: ``x`` = channel, ``y`` = width, ``z`` = height. Shared edges
    are the same 1-D vectors on adjacent faces.
    """
    arr = np.asarray(vol, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {arr.shape}")
    return {
        "xm": np.asarray(arr[0], dtype=float),
        "xp": np.asarray(arr[-1], dtype=float),
        "ym": np.asarray(arr[:, :, 0].T, dtype=float),
        "yp": np.asarray(arr[:, :, -1].T, dtype=float),
        "zm": np.asarray(arr[:, 0, :].T, dtype=float),
        "zp": np.asarray(arr[:, -1, :].T, dtype=float),
    }


def fourier_basis_plane(kx: int, ky: int, kind: str, *, size: int = 128) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, size, dtype=np.float32),
        np.linspace(0.0, 2.0 * np.pi, size, dtype=np.float32),
        indexing="ij",
    )
    if kind == "const":
        return np.ones((size, size), dtype=np.float32)
    phase = float(kx) * xx + float(ky) * yy
    if kind == "cos":
        return np.cos(phase).astype(np.float32)
    if kind == "sin":
        return np.sin(phase).astype(np.float32)
    raise ValueError(f"unknown Fourier kind {kind!r}")


def enumerate_illustration_modes(n_modes: int) -> tuple[tuple[int, int, str], ...]:
    """Same ordering as ``enumerate_fourier_modes`` (no torch)."""
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1; got {n_modes}")
    modes: list[tuple[int, int, str]] = []
    modes.append((0, 0, "const"))
    degree = 1
    while len(modes) < n_modes:
        for kx in range(degree, -1, -1):
            ky = degree - kx
            if kx == 0 and ky == 0:
                continue
            modes.append((kx, ky, "cos"))
            if len(modes) >= n_modes:
                break
            modes.append((kx, ky, "sin"))
            if len(modes) >= n_modes:
                break
        degree += 1
    return tuple(modes[:n_modes])


def fourier_term_multiplied(feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``feat[c] * basis[c]`` and channel-wise spatial mean (coded pool)."""
    arr = np.asarray(feat, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {arr.shape}")
    n_ch, height, _width = arr.shape
    modes = enumerate_illustration_modes(n_ch)
    pre = np.empty_like(arr)
    for index, (kx, ky, kind) in enumerate(modes):
        pre[index] = arr[index] * fourier_basis_plane(
            kx, ky, kind, size=height
        )
    pooled = pre.mean(axis=(1, 2))
    return pre, pooled


def scalar_rgb(value: float, *, vmin: float, vmax: float, cmap: str = "turbo") -> tuple[float, float, float]:
    if vmax - vmin < 1e-12:
        t = 0.5
    else:
        t = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    rgba = colormaps[cmap](t)
    return float(rgba[0]), float(rgba[1]), float(rgba[2])
