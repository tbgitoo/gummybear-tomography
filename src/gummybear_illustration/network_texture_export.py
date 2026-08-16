"""Write all network-illustration texture PNGs next to the ``.pov`` file."""

from __future__ import annotations

from pathlib import Path

from .network_activations import NetworkActivationBundle
from .network_textures import (
    channel_colormap_limits,
    enumerate_illustration_modes,
    volume_boundary_faces,
    write_colormap_png,
    write_face_png_per_channel,
    write_fourier_pane_png,
    write_vector_strip_png,
    write_zscore_gray_png,
)

_FACE_KEYS = ("xm", "xp", "ym", "yp", "zm", "zp")


def _write_volume_faces(
    vol,
    dest: Path,
    *,
    stem: str,
    tag: str,
    scale: str = "minmax",
    zscore_clip: float = 2.0,
) -> dict[str, str]:
    """Per-volume, per-channel scale (never shared across Fourier vs GAP)."""
    vmin, vmax = channel_colormap_limits(vol)
    faces = volume_boundary_faces(vol)
    axes: dict[str, int | None] = {
        "xm": None,
        "xp": None,
        "ym": 1,
        "yp": 1,
        "zm": 1,
        "zp": 1,
    }
    names: dict[str, str] = {}
    for key in _FACE_KEYS:
        if key == "xm":
            lo, hi = vmin[0], vmax[0]
        elif key == "xp":
            lo, hi = vmin[-1], vmax[-1]
        else:
            lo, hi = vmin, vmax
        names[key] = write_face_png_per_channel(
            faces[key],
            dest / f"{stem}_{tag}_{key}.png",
            channel_axis=axes[key],
            vmin=lo,
            vmax=hi,
            scale=scale,
            zscore_clip=zscore_clip,
        )
    return names


def write_network_texture_pngs(
    bundle: NetworkActivationBundle,
    directory: str | Path,
    *,
    stem: str,
    zscore_clip: float,
    n_fourier_planes: int = 8,
    n_input_stack: int = 1,
    slice_colormap: str = "minmax",
    slice_colormap_clip: float = 2.0,
) -> dict[str, str | list[str] | list[dict[str, str]]]:
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    names: dict[str, str | list[str] | list[dict[str, str]]] = {}
    gray = write_zscore_gray_png(
        bundle.input_zscore,
        dest / f"{stem}_input_zscore.png",
        clip=zscore_clip,
    )
    slice_cm = str(slice_colormap).strip().lower()
    slice_clip = float(slice_colormap_clip)
    names["input"] = [gray] * int(n_input_stack)
    cnn: list[dict[str, str]] = []
    for vol, tag in zip(bundle.conv_maps, ("c16", "c32", "c64"), strict=True):
        cnn.append(
            _write_volume_faces(
                vol,
                dest,
                stem=stem,
                tag=tag,
                scale=slice_cm,
                zscore_clip=slice_clip,
            )
        )
    names["cnn"] = cnn
    cnn_gap: list[dict[str, str]] = []
    for vol, tag in zip(bundle.gap_conv_maps, ("c16", "c32", "c64"), strict=True):
        cnn_gap.append(
            _write_volume_faces(
                vol,
                dest,
                stem=stem,
                tag=f"gap_{tag}",
                scale=slice_cm,
                zscore_clip=slice_clip,
            )
        )
    names["cnn_gap"] = cnn_gap
    feat = bundle.conv_maps[-1]
    n_planes = max(int(n_fourier_planes), 1)
    modes = enumerate_illustration_modes(n_planes)
    fourier: list[str] = []
    for i, _mode in enumerate(modes[:n_planes]):
        ch = int(i % feat.shape[0])
        fourier.append(
            write_colormap_png(
                feat[ch],
                dest / f"{stem}_fourier_{i:02d}.png",
                scale=slice_cm,
                zscore_clip=slice_clip,
            )
        )
    names["fourier"] = fourier
    names["flatten"] = _write_volume_faces(
        feat,
        dest,
        stem=stem,
        tag="flatten",
        scale=slice_cm,
        zscore_clip=slice_clip,
    )
    names["gap"] = write_vector_strip_png(
        bundle.gap, dest / f"{stem}_gap_pool.png"
    )
    names["fourier_pane"] = write_fourier_pane_png(
        bundle.fourier_prepool,
        dest / f"{stem}_fourier_pane.png",
        scale=slice_cm,
        zscore_clip=slice_clip,
    )
    names["gap_pane"] = write_fourier_pane_png(
        bundle.gap_prepool,
        dest / f"{stem}_gap_pane.png",
        scale=slice_cm,
        zscore_clip=slice_clip,
    )
    names["fourier_pool"] = write_vector_strip_png(
        bundle.fourier_pooled, dest / f"{stem}_fourier_pool.png"
    )
    return names
