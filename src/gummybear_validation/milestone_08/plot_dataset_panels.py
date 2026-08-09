"""Dataset validation panels for raw float absolute-intensity views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from tomography_ml.gummybear_data_catalog.catalog import CatalogRow, ParticleLabel
from tomography_ml.gummybear_data_catalog.task_dataset import load_role_array


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load one sequence ``manifest.json`` as a Python dict.

    Args:
        manifest_path: Path to ``manifest.json``.

    Returns:
        dict: Parsed manifest object.

    Raises:
        ValueError: When the JSON root is not an object.
    """
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest is not a JSON object: {path}")
    return payload


def project_world_points_to_pixels(
    points_xyz: np.ndarray,
    *,
    camera_position: Sequence[float],
    look_at: Sequence[float],
    up: Sequence[float],
    fov_deg: float,
    resolution: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with the pinhole convention used by ``make_pinhole_rays``.

    Returns pixel coordinates as ``(col, row)`` matching ``imshow`` raster order,
    plus a boolean mask for points in front of the camera (positive depth).

    Args:
        points_xyz: World points, shape ``(N, 3)`` or ``(3,)``.
        camera_position: Camera centre in world coordinates.
        look_at: Look-at point defining view direction.
        up: Up vector hint for camera basis construction.
        fov_deg: Vertical field of view in degrees.
        resolution: ``(height, width)`` image size from the manifest.

    Returns:
        tuple: ``(uv, in_front)`` where ``uv`` has shape ``(N, 2)`` and
        ``in_front`` has shape ``(N,)``.

    Raises:
        ValueError: When ``points_xyz`` is not ``(N, 3)``.
    """
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N, 3); got {points.shape}")

    camera_pos = np.asarray(camera_position, dtype=float).reshape(3)
    look = np.asarray(look_at, dtype=float).reshape(3)
    up_hint = np.asarray(up, dtype=float).reshape(3)

    forward = look - camera_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    cam_up = cam_up / np.linalg.norm(cam_up)

    height = int(resolution[0])
    width = int(resolution[1]) if len(resolution) > 1 else height
    half_size = float(np.tan(np.deg2rad(float(fov_deg)) / 2.0))

    relative = points - camera_pos
    depth = relative @ forward
    in_front = depth > 1e-8
    # Intersection with the unit-distance image plane used by make_pinhole_rays.
    t = np.ones(len(points), dtype=float)
    t[in_front] = 1.0 / depth[in_front]
    plane_points = camera_pos + t[:, None] * relative
    offset = plane_points - (camera_pos + forward)
    u = offset @ right
    v = offset @ cam_up

    col = (u + half_size) / (2.0 * half_size) * (width - 1)
    row = (v + half_size) / (2.0 * half_size) * (height - 1)
    return np.column_stack([col, row]), in_front


def _chw_to_hw(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        if arr.shape[0] == 1:
            return arr[0]
        return np.mean(arr, axis=0)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return np.mean(arr, axis=-1)
    raise ValueError(f"Unsupported image shape for display: {arr.shape}")


def approximate_radius_pixels(
    *,
    radius_world: float,
    depth: float,
    fov_deg: float,
    resolution: Sequence[int],
) -> float:
    """Estimate on-axis pixel radius for a sphere of given world radius.

    Uses a pinhole approximation at ``depth`` along the view axis. Returns
    ``0.0`` when depth or field of view is degenerate.

    Args:
        radius_world: Physical sphere radius in world units.
        depth: Distance from camera to sphere centre along view.
        fov_deg: Vertical field of view in degrees.
        resolution: ``(height, width)`` image size.

    Returns:
        float: Approximate radius in pixels (not clamped).
    """
    height = int(resolution[0])
    half_size = float(np.tan(np.deg2rad(float(fov_deg)) / 2.0))
    if depth <= 1e-8 or half_size <= 0.0:
        return 0.0
    return float(radius_world / depth / (2.0 * half_size) * (height - 1))


def _imshow_absolute(
    axis: Axes,
    image_hw: np.ndarray,
    *,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "gray",
) -> Any:
    handle = axis.imshow(
        image_hw,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.axis("off")
    return handle


def plot_sequence_role_panel(
    row: CatalogRow,
    *,
    frame_index: int = 0,
    figsize: tuple[float, float] = (16.0, 3.4),
) -> Figure:
    """Plot observed / clean / particle / delta / overlay for one catalog frame.

    Loads **raw float** role arrays (authoritative sidecars). Observed, clean,
    and particle share one absolute intensity scale; delta uses a symmetric
    scale about zero. The overlay panel projects particle centres from manifest
    camera geometry.

    Args:
        row: Catalog row with role refs and particle labels.
        frame_index: View index ``0 … V−1``.
        figsize: Matplotlib figure size in inches.

    Returns:
        matplotlib.figure.Figure: Five-panel figure (not shown).

    Raises:
        ValueError: When required role refs are missing.
        IndexError: When ``frame_index`` is out of range.
    """
    if row.observed_ref is None or row.clean_ref is None or row.particle_ref is None:
        raise ValueError(
            f"Catalog row {row.sequence_id!r} is missing required role refs "
            "(observed/clean/particle)."
        )

    observed = load_role_array(row.observed_ref)  # raw_float default
    clean = load_role_array(row.clean_ref)
    particle = load_role_array(row.particle_ref)
    if not (0 <= frame_index < observed.shape[0]):
        raise IndexError(
            f"frame_index={frame_index} out of range for V={observed.shape[0]}"
        )

    obs_hw = _chw_to_hw(observed[frame_index])
    clean_hw = _chw_to_hw(clean[frame_index])
    particle_hw = _chw_to_hw(particle[frame_index])
    delta_hw = clean_hw - obs_hw

    abs_max = float(
        np.max(
            [
                np.max(np.abs(obs_hw)),
                np.max(np.abs(clean_hw)),
                np.max(np.abs(particle_hw)),
            ]
        )
    )
    vmax = abs_max if abs_max > 0.0 else 1.0
    delta_max = float(np.max(np.abs(delta_hw)))
    delta_lim = delta_max if delta_max > 0.0 else 1.0

    fig, axes = plt.subplots(1, 5, figsize=figsize)
    _imshow_absolute(axes[0], obs_hw, title="observed", vmin=0.0, vmax=vmax)
    _imshow_absolute(axes[1], clean_hw, title="clean", vmin=0.0, vmax=vmax)
    _imshow_absolute(axes[2], particle_hw, title="particle", vmin=0.0, vmax=vmax)
    im_delta = _imshow_absolute(
        axes[3],
        delta_hw,
        title="float delta (clean−observed)",
        vmin=-delta_lim,
        vmax=delta_lim,
        cmap="coolwarm",
    )
    fig.colorbar(im_delta, ax=axes[3], fraction=0.046, pad=0.04)

    _imshow_absolute(
        axes[4],
        obs_hw,
        title="particle target overlay",
        vmin=0.0,
        vmax=vmax,
    )
    _draw_particle_overlays(axes[4], row=row, frame_index=frame_index)

    angle = (
        row.angles_deg[frame_index]
        if frame_index < len(row.angles_deg)
        else float("nan")
    )
    fig.suptitle(
        f"{row.sequence_id}  frame={frame_index}  angle={angle:g}°  "
        f"μa={row.bear_mu_a:g}  μs={row.bear_mu_s:g}  "
        f"vmax={vmax:.4g}  (raw float)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _draw_particle_overlays(
    axis: Axes,
    *,
    row: CatalogRow,
    frame_index: int,
) -> None:
    manifest = load_manifest(row.manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or frame_index >= len(frames):
        return
    frame = frames[frame_index]
    if not isinstance(frame, dict):
        return

    labels: Sequence[ParticleLabel] = row.particles
    if not labels:
        return

    centres = np.asarray(
        [[p.center_x, p.center_y, p.center_z] for p in labels],
        dtype=float,
    )
    uv, in_front = project_world_points_to_pixels(
        centres,
        camera_position=frame["camera_position"],
        look_at=frame["look_at"],
        up=frame["up"],
        fov_deg=float(frame["fov_deg"]),
        resolution=frame["resolution"],
    )
    camera_pos = np.asarray(frame["camera_position"], dtype=float)
    for particle, (col, row_px), visible in zip(labels, uv, in_front, strict=True):
        if not visible:
            continue
        centre = np.asarray(
            [particle.center_x, particle.center_y, particle.center_z],
            dtype=float,
        )
        depth = float(np.linalg.norm(centre - camera_pos))
        radius_px = approximate_radius_pixels(
            radius_world=float(particle.radius),
            depth=depth,
            fov_deg=float(frame["fov_deg"]),
            resolution=frame["resolution"],
        )
        axis.plot(col, row_px, marker="+", color="cyan", markersize=10, mew=1.5)
        axis.add_patch(
            Circle(
                (col, row_px),
                radius=max(radius_px, 2.0),
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
            )
        )


def plot_regime_intensity_histograms(
    rows_by_regime: Mapping[str, CatalogRow],
    *,
    role: str = "observed",
    bins: int = 64,
    figsize: tuple[float, float] = (8.0, 4.5),
    xlim_quantile: float = 0.995,
    xlim_pad: float = 0.05,
) -> Figure:
    """Histogram raw-float intensities for low / medium / high optical regimes.

    ``rows_by_regime`` maps regime label → one representative ``CatalogRow``.
    Pools all views; excludes zero pixels (background / missed rays). X-axis
    is clipped to the pooled ``xlim_quantile`` so hot outliers do not flatten
    the bulk near zero.

    Args:
        rows_by_regime: Mapping of regime name to catalog row.
        role: Role to load (default ``observed``).
        bins: Histogram bin count between clipped min and max.
        figsize: Figure size in inches.
        xlim_quantile: Upper quantile for x-axis clip (default 99.5%).
        xlim_pad: Multiplicative padding above the clip high.

    Returns:
        matplotlib.figure.Figure: Single-axis density histogram.

    Raises:
        ValueError: When quantile/pad invalid or no positive finite samples exist.
    """
    if not 0.0 < float(xlim_quantile) <= 1.0:
        raise ValueError(f"xlim_quantile must be in (0, 1], got {xlim_quantile}")
    if float(xlim_pad) < 0.0:
        raise ValueError(f"xlim_pad must be >= 0, got {xlim_pad}")

    series: list[tuple[str, CatalogRow, np.ndarray]] = []
    for regime, row in rows_by_regime.items():
        ref = getattr(row, f"{role}_ref", None)
        if ref is None:
            raise ValueError(f"Row {row.sequence_id!r} has no {role}_ref")
        array = load_role_array(ref)
        values = np.asarray(array, dtype=np.float64).ravel()
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size == 0:
            raise ValueError(
                f"No positive finite {role} samples for {row.sequence_id!r}"
            )
        series.append((regime, row, values))

    pooled = np.concatenate([values for _, _, values in series])
    if pooled.size == 0:
        raise ValueError("No positive finite intensity samples to histogram")
    x_hi = float(np.quantile(pooled, xlim_quantile))
    x_max = float(pooled.max())
    if x_hi <= 0.0:
        x_hi = x_max if x_max > 0.0 else 1.0
    x_hi *= 1.0 + float(xlim_pad)
    # Start bins at the pooled positive minimum so the empty [0, min] gap
    # does not dominate the left side after dropping zeros.
    x_lo = float(pooled.min())
    if x_lo >= x_hi:
        x_lo = 0.0
    bin_edges = np.linspace(x_lo, x_hi, int(bins) + 1)

    fig, axis = plt.subplots(1, 1, figsize=figsize)
    summaries: list[str] = []
    for regime, row, values in series:
        axis.hist(
            values,
            bins=bin_edges,
            histtype="step",
            density=True,
            label=(
                f"{regime}  {row.sequence_id}  "
                f"μa={row.bear_mu_a:g} μs={row.bear_mu_s:g}"
            ),
        )
        summaries.append(
            f"{regime}: mean={values.mean():.4g}  "
            f"p95={np.quantile(values, 0.95):.4g}  "
            f"max={values.max():.4g}"
        )

    axis.set_xlim(x_lo, x_hi)
    axis.set_xlabel(
        f"{role} absolute intensity (raw float; zeros excluded; "
        f"xlim=p{xlim_quantile * 100:g}, max={x_max:.4g})"
    )
    axis.set_ylabel("density")
    axis.set_title("Optical-regime absolute intensity (no auto-contrast)")
    axis.legend(loc="best", fontsize=8)
    axis.text(
        0.02,
        0.98,
        "\n".join(summaries),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    return fig


def plot_regime_role_grid(
    rows_by_regime: Mapping[str, CatalogRow],
    *,
    frame_index: int = 0,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot one row per regime: observed / clean / particle / delta (shared vmax).

    Args:
        rows_by_regime: Mapping of regime name to catalog row.
        frame_index: View index shared across rows.
        figsize: Optional figure size; defaults to ``(14, 2.8 * n_rows)``.

    Returns:
        matplotlib.figure.Figure: ``n_regimes × 4`` panel grid.

    Raises:
        ValueError: When mapping empty or a row lacks required role refs.
    """
    regime_items = list(rows_by_regime.items())
    if not regime_items:
        raise ValueError("rows_by_regime is empty")

    loaded: list[tuple[str, CatalogRow, np.ndarray, np.ndarray, np.ndarray]] = []
    global_vmax = 0.0
    global_delta = 0.0
    for regime, row in regime_items:
        if row.observed_ref is None or row.clean_ref is None or row.particle_ref is None:
            raise ValueError(f"Incomplete roles for {row.sequence_id!r}")
        observed = _chw_to_hw(load_role_array(row.observed_ref)[frame_index])
        clean = _chw_to_hw(load_role_array(row.clean_ref)[frame_index])
        particle = _chw_to_hw(load_role_array(row.particle_ref)[frame_index])
        global_vmax = max(
            global_vmax,
            float(np.max(observed)),
            float(np.max(clean)),
            float(np.max(particle)),
        )
        delta = clean - observed
        global_delta = max(global_delta, float(np.max(np.abs(delta))))
        loaded.append((regime, row, observed, clean, particle))

    vmax = global_vmax if global_vmax > 0.0 else 1.0
    delta_lim = global_delta if global_delta > 0.0 else 1.0
    n_rows = len(loaded)
    if figsize is None:
        figsize = (14.0, 2.8 * n_rows)

    fig, axes = plt.subplots(n_rows, 4, figsize=figsize, squeeze=False)
    for row_i, (regime, row, observed, clean, particle) in enumerate(loaded):
        delta = clean - observed
        titles = (
            f"{regime} observed",
            f"{regime} clean",
            f"{regime} particle",
            f"{regime} Δ",
        )
        images = (observed, clean, particle, delta)
        cmaps = ("gray", "gray", "gray", "coolwarm")
        limits = (
            (0.0, vmax),
            (0.0, vmax),
            (0.0, vmax),
            (-delta_lim, delta_lim),
        )
        for col_i, (title, image, cmap, (vmin, vmax_i)) in enumerate(
            zip(titles, images, cmaps, limits, strict=True)
        ):
            _imshow_absolute(
                axes[row_i, col_i],
                image,
                title=title,
                vmin=vmin,
                vmax=vmax_i,
                cmap=cmap,
            )
        axes[row_i, 0].set_ylabel(
            f"{row.sequence_id}\nμa={row.bear_mu_a:g}",
            fontsize=9,
        )

    fig.suptitle(
        f"Shared absolute scale across regimes  vmax={vmax:.4g}  "
        f"(raw float, frame={frame_index})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def write_sequence_role_orbit_gif(
    row: CatalogRow,
    output_path: str | Path,
    *,
    role: str = "observed",
    fps: float = 8.0,
    vmax_quantile: float = 0.995,
    annotate_angles: bool = True,
) -> Path:
    """Write an ordered multi-view GIF for one catalog role (raw float scale).

    Maps intensity once per sequence with ``vmin=0`` and ``vmax`` at
    ``vmax_quantile`` so frames are not per-view auto-contrasted. Optional
    angle annotations require Pillow (core dependency).

    Args:
        row: Catalog row with ``angles_deg`` and role ref.
        output_path: Destination ``.gif`` path (parent dirs created).
        role: Role name suffix for ``{role}_ref`` lookup.
        fps: Playback frames per second.
        vmax_quantile: Pooled quantile defining shared display max.
        annotate_angles: If True, burn sequence/frame/angle text into frames.

    Returns:
        pathlib.Path: Resolved output path written.

    Raises:
        ValueError: When role ref missing or array is not ``(V, C, H, W)``.
    """
    from PIL import Image, ImageDraw

    ref = getattr(row, f"{role}_ref", None)
    if ref is None:
        raise ValueError(f"Row {row.sequence_id!r} has no {role}_ref")
    array = load_role_array(ref)
    if array.ndim != 4:
        raise ValueError(f"Expected (V,C,H,W); got {array.shape}")
    frames_hw = np.stack([_chw_to_hw(array[i]) for i in range(array.shape[0])])
    finite = frames_hw[np.isfinite(frames_hw)]
    if finite.size == 0:
        raise ValueError(f"No finite pixels in {role} for {row.sequence_id!r}")
    vmax = float(np.quantile(np.maximum(finite, 0.0), vmax_quantile))
    if vmax <= 0.0:
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    scaled = np.clip(frames_hw / vmax, 0.0, 1.0)
    uint8 = (scaled * 255.0).astype(np.uint8)

    angles = row.angles_deg
    images: list[Image.Image] = []
    for index, plane in enumerate(uint8):
        img = Image.fromarray(plane, mode="L").convert("RGB")
        if annotate_angles and index < len(angles):
            draw = ImageDraw.Draw(img)
            label = (
                f"{row.sequence_id}  {role}  "
                f"frame={index:02d}  angle={angles[index]:.1f}°  "
                f"μa={row.bear_mu_a:g}"
            )
            draw.rectangle((0, 0, img.width, 14), fill=(0, 0, 0))
            draw.text((2, 1), label, fill=(255, 255, 255))
        images.append(img)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(round(1000.0 / float(fps))), 1)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return out
