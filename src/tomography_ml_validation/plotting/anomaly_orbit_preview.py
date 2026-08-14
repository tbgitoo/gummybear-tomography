"""Still-vs-orbit anomaly previews for final-report / notebook illustration.

Display-only helpers: fixed linear mean/std greyscale mapping and an in-memory
GIF for multi-view orbits. Not the ML training normalisation path
(``per_image_zscore``).
"""

from __future__ import annotations

import html as html_lib
from base64 import b64encode
from io import BytesIO
from typing import Sequence

import numpy as np
from IPython.display import HTML, display
from PIL import Image as PILImage, ImageDraw

# Illustration defaults (linear anomaly intensity). Tuned so a dim 0° still
# stays readable while brighter orbit angles may saturate locally.
DEFAULT_DISPLAY_MEAN = 8.0e-5
DEFAULT_DISPLAY_STD = 1.0e-4
DEFAULT_DISPLAY_PX = 256
DEFAULT_FPS = 8.0
# Cam × light grid GIFs have more frames; keep playback readable.
DEFAULT_CAMERA_LIGHT_GRID_FPS = 2.0


def _as_hw(plane: np.ndarray) -> np.ndarray:
    """Accept ``(H, W)`` or ``(1, H, W)`` / ``(C, H, W)`` with ``C=1``."""
    arr = np.asarray(plane, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(
        f"Expected still plane (H, W) or (1, H, W); got shape {tuple(arr.shape)}"
    )


def _as_vhw(stack: np.ndarray) -> np.ndarray:
    """Accept ``(V, H, W)`` or ``(V, 1, H, W)``."""
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(
        f"Expected orbit stack (V, H, W) or (V, 1, H, W); got shape {tuple(arr.shape)}"
    )


def anomaly_plane_to_rgb(
    plane: np.ndarray,
    *,
    mean: float,
    std: float,
    display_px: int = DEFAULT_DISPLAY_PX,
) -> PILImage.Image:
    """Map one signed anomaly plane to upright RGB with fixed display scale.

    Mid-gray at ``mean``; black/white at ``mean ± std``. Values outside that
    range clip in the 8-bit conversion. Nearest-neighbour upsample to
    ``display_px``.
    """
    if std <= 0.0:
        raise ValueError(f"std must be > 0; got {std}")
    if display_px < 1:
        raise ValueError(f"display_px must be >= 1; got {display_px}")
    hw = _as_hw(plane)
    scaled = np.clip(0.5 + 0.5 * ((hw - float(mean)) / float(std)), 0.0, 1.0)
    uint8 = (scaled * 255.0).astype(np.uint8)
    return (
        PILImage.fromarray(uint8, mode="L")
        .resize((int(display_px), int(display_px)), resample=PILImage.Resampling.NEAREST)
        .convert("RGB")
    )


def _annotate(frame: PILImage.Image, text: str) -> None:
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, frame.width, 14), fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 255))


def build_anomaly_still_png(
    still_hw: np.ndarray,
    *,
    label: str,
    mean: float = DEFAULT_DISPLAY_MEAN,
    std: float = DEFAULT_DISPLAY_STD,
    display_px: int = DEFAULT_DISPLAY_PX,
) -> bytes:
    """Return PNG bytes for one annotated still frame."""
    image = anomaly_plane_to_rgb(still_hw, mean=mean, std=std, display_px=display_px)
    _annotate(image, label)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def build_anomaly_orbit_gif(
    orbit_vhw: np.ndarray,
    *,
    sequence_id: str,
    angles_deg: Sequence[float],
    mean: float = DEFAULT_DISPLAY_MEAN,
    std: float = DEFAULT_DISPLAY_STD,
    display_px: int = DEFAULT_DISPLAY_PX,
    fps: float = DEFAULT_FPS,
) -> bytes:
    """Return GIF bytes for an annotated multi-view anomaly orbit."""
    stack = _as_vhw(orbit_vhw)
    n_views = int(stack.shape[0])
    if n_views < 1:
        raise ValueError("orbit stack must contain at least one view")
    angles = tuple(float(a) for a in angles_deg)
    duration_ms = max(int(round(1000.0 / float(fps))), 1)
    frames: list[PILImage.Image] = []
    for index in range(n_views):
        frame = anomaly_plane_to_rgb(
            stack[index], mean=mean, std=std, display_px=display_px
        )
        angle = angles[index] if index < len(angles) else float(index)
        _annotate(
            frame,
            f"M9 orbit  {sequence_id}  frame={index:02d}  angle={angle:.0f}°",
        )
        frames.append(frame)
    buf = BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()


def _panel_heading_html(heading: str | None) -> str:
    """Larger panel heading, or empty if heading is omitted."""
    if heading is None or not str(heading).strip():
        return ""
    return (
        "<div style='font: 600 18px sans-serif; margin-bottom:0.2rem;'>"
        f"{html_lib.escape(str(heading).strip())}</div>"
    )


def _panel_caption_html(caption: str | None) -> str:
    """12px technical title under the heading, or empty if omitted."""
    if caption is None or not str(caption).strip():
        return ""
    return (
        "<div style='font: 12px sans-serif; margin-bottom:0.35rem;'>"
        f"{html_lib.escape(str(caption).strip())}</div>"
    )


def anomaly_still_vs_orbit_html(
    still_hw: np.ndarray,
    orbit_vhw: np.ndarray,
    *,
    still_sequence_id: str,
    orbit_sequence_id: str,
    still_angle_deg: float,
    orbit_angles_deg: Sequence[float],
    mean: float = DEFAULT_DISPLAY_MEAN,
    std: float = DEFAULT_DISPLAY_STD,
    display_px: int = DEFAULT_DISPLAY_PX,
    fps: float = DEFAULT_FPS,
    still_caption: str | None = None,
    orbit_caption: str | None = None,
    still_heading: str | None = None,
    orbit_heading: str | None = None,
) -> HTML:
    """Build side-by-side HTML for an M8 still and M9 orbit GIF.

    Args:
        still_hw: Single-view anomaly ``(H, W)`` (or ``(1, H, W)``).
        orbit_vhw: Multi-view anomaly ``(V, H, W)`` (or ``(V, 1, H, W)``).
        still_sequence_id / orbit_sequence_id: Labels burned into frames.
        still_angle_deg: Acquisition angle for the still overlay text.
        orbit_angles_deg: Per-view angles for GIF frame labels.
        mean, std: Fixed display affine (not ML per-view z-score).
        display_px: Shared on-screen size (nearest upsample).
        fps: GIF playback rate.
        still_caption / orbit_caption: Technical titles (12px) above each panel.
            If omitted, a shape/layout string is generated.
        still_heading / orbit_heading: Larger panel headings; ``None`` or ``""``
            omits them. Pass from the notebook for report figures.

    Returns:
        ``IPython.display.HTML`` suitable for ``display(...)``.
    """
    stack = _as_vhw(orbit_vhw)
    n_views = int(stack.shape[0])
    h = int(_as_hw(still_hw).shape[0])
    w = int(_as_hw(still_hw).shape[1])

    still_label = (
        f"M8 single view  {still_sequence_id}  angle={float(still_angle_deg):.0f}°"
    )
    png_bytes = build_anomaly_still_png(
        still_hw,
        label=still_label,
        mean=mean,
        std=std,
        display_px=display_px,
    )
    gif_bytes = build_anomaly_orbit_gif(
        stack,
        sequence_id=orbit_sequence_id,
        angles_deg=orbit_angles_deg,
        mean=mean,
        std=std,
        display_px=display_px,
        fps=fps,
    )
    png_b64 = b64encode(png_bytes).decode("ascii")
    gif_b64 = b64encode(gif_bytes).decode("ascii")

    left_title = still_caption or (
        f"M8 [1,1,{h},{w}]; V=1(still), C=1(greyscale)"
    )
    right_title = orbit_caption or (
        f"M9 [{n_views},1,{h},{w}]; V={n_views}(orbit), C=1(greyscale)"
    )
    return HTML(
        "<div style='display:flex;gap:1.25rem;align-items:flex-start;'>"
        "<div style='text-align:center;'>"
        f"{_panel_heading_html(still_heading)}"
        f"{_panel_caption_html(left_title)}"
        f"<img width='{int(display_px)}' height='{int(display_px)}' "
        f"src='data:image/png;base64,{png_b64}'/>"
        "</div>"
        "<div style='text-align:center;'>"
        f"{_panel_heading_html(orbit_heading)}"
        f"{_panel_caption_html(right_title)}"
        f"<img width='{int(display_px)}' height='{int(display_px)}' "
        f"src='data:image/gif;base64,{gif_b64}'/>"
        "</div>"
        "</div>"
        "<p style='font: 12px sans-serif; max-width: 38rem;'>"
        "<b>Display only (fixed scale):</b> "
        f"both panels use mean={float(mean):g}, std={float(std):.0e}; "
        "8-bit conversion maps ±1 on that scale to black/white (outside clips). "
        "Chosen for illustration so the dim 0° still stays readable while "
        "brighter orbit angles may saturate locally. ML training instead uses "
        "per-image (per-view) z-score."
        "</p>"
    )


def display_anomaly_still_vs_orbit(
    still_hw: np.ndarray,
    orbit_vhw: np.ndarray,
    *,
    still_sequence_id: str,
    orbit_sequence_id: str,
    still_angle_deg: float,
    orbit_angles_deg: Sequence[float],
    mean: float = DEFAULT_DISPLAY_MEAN,
    std: float = DEFAULT_DISPLAY_STD,
    display_px: int = DEFAULT_DISPLAY_PX,
    fps: float = DEFAULT_FPS,
    still_caption: str | None = None,
    orbit_caption: str | None = None,
    still_heading: str | None = None,
    orbit_heading: str | None = None,
) -> HTML:
    """Build and ``display`` the still-vs-orbit anomaly HTML preview.

    Returns the same ``HTML`` object for callers that want to capture it.
    """
    html = anomaly_still_vs_orbit_html(
        still_hw,
        orbit_vhw,
        still_sequence_id=still_sequence_id,
        orbit_sequence_id=orbit_sequence_id,
        still_angle_deg=still_angle_deg,
        orbit_angles_deg=orbit_angles_deg,
        mean=mean,
        std=std,
        display_px=display_px,
        fps=fps,
        still_caption=still_caption,
        orbit_caption=orbit_caption,
        still_heading=still_heading,
        orbit_heading=orbit_heading,
    )
    display(html)
    return html


def _as_illum_view_hw(grid: np.ndarray) -> np.ndarray:
    """Accept canonical ``[I, V, C, H, W]`` (``C=1``) or ``[I, V, H, W]``."""
    arr = np.asarray(grid, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.ndim == 4:
        return arr
    raise ValueError(
        "Expected illumination-major grid [I, V, C, H, W] with C=1 "
        f"or [I, V, H, W]; got shape {tuple(arr.shape)}"
    )


def build_anomaly_camera_light_grid_gif(
    grid_iv: np.ndarray,
    *,
    sequence_id: str,
    camera_angles_deg: Sequence[float],
    light_angles_deg: Sequence[float],
    mean: float = DEFAULT_DISPLAY_MEAN,
    std: float = DEFAULT_DISPLAY_STD,
    display_px: int = DEFAULT_DISPLAY_PX,
    fps: float = DEFAULT_CAMERA_LIGHT_GRID_FPS,
) -> bytes:
    """GIF sweeping canonical ``[I, V]`` anomaly grid (illumination-major).

    Frame order matches ``views[light_i, cam_j]``: for each illumination, all
    camera views, then the next light. Playback defaults slower than M9 orbits.
    """
    grid = _as_illum_view_hw(grid_iv)
    n_light, n_cam = int(grid.shape[0]), int(grid.shape[1])
    if n_cam < 1 or n_light < 1:
        raise ValueError("I×V grid must be non-empty")
    if not str(sequence_id):
        raise ValueError("sequence_id must be non-empty")
    cam_angles = tuple(float(a) for a in camera_angles_deg)
    light_angles = tuple(float(a) for a in light_angles_deg)
    duration_ms = max(int(round(1000.0 / float(fps))), 1)
    frames: list[PILImage.Image] = []
    for light_i in range(n_light):
        light_deg = (
            light_angles[light_i] if light_i < len(light_angles) else float(light_i)
        )
        for cam_j in range(n_cam):
            cam_deg = cam_angles[cam_j] if cam_j < len(cam_angles) else float(cam_j)
            frame = anomaly_plane_to_rgb(
                grid[light_i, cam_j], mean=mean, std=std, display_px=display_px
            )
            # Put indices/angles first so GIF frames stay distinct when the
            # annotate strip would otherwise clip a long sequence id.
            _annotate(
                frame,
                f"[{light_i},{cam_j}] light={light_deg:.0f}° cam={cam_deg:.0f}°",
            )
            frames.append(frame)
    buf = BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return buf.getvalue()


def anomaly_camera_light_grid_html(
    still_hw: np.ndarray | None,
    grid_iv: np.ndarray,
    *,
    sequence_id: str,
    camera_angles_deg: Sequence[float],
    light_angles_deg: Sequence[float],
    still_camera_deg: float | None = None,
    still_light_deg: float | None = None,
    mean: float = DEFAULT_DISPLAY_MEAN,
    std: float = DEFAULT_DISPLAY_STD,
    display_px: int = DEFAULT_DISPLAY_PX,
    fps: float = DEFAULT_CAMERA_LIGHT_GRID_FPS,
    include_still: bool = True,
    heading: str | None = None,
    caption: str | None = None,
    still_heading: str | None = None,
    still_caption: str | None = None,
) -> HTML:
    """GIF over the full ``[I, V]`` grid (canonical M10 sample); optional still.

    Display-only fixed mean/std greyscale; not ML per-view z-score.

    Args:
        heading / caption: GIF-panel large heading and 12px technical title.
            Pass from the notebook for report figures. If ``caption`` is
            omitted, a shape/layout string is generated.
        still_heading / still_caption: Same for the optional still panel.
    """
    grid = _as_illum_view_hw(grid_iv)
    n_light, n_cam = int(grid.shape[0]), int(grid.shape[1])
    h, w = int(grid.shape[-2]), int(grid.shape[-1])
    n_frames = n_cam * n_light

    gif_bytes = build_anomaly_camera_light_grid_gif(
        grid,
        sequence_id=sequence_id,
        camera_angles_deg=camera_angles_deg,
        light_angles_deg=light_angles_deg,
        mean=mean,
        std=std,
        display_px=display_px,
        fps=fps,
    )
    gif_b64 = b64encode(gif_bytes).decode("ascii")
    gif_caption = caption or (
        f"M10 sample [{n_light},{n_cam},1,{h},{w}] = [I,V,C,H,W]  "
        f"GIF sweeps {n_frames} frames @ {float(fps):g} fps"
    )
    gif_panel = (
        "<div style='text-align:center;'>"
        f"{_panel_heading_html(heading)}"
        f"{_panel_caption_html(gif_caption)}"
        f"<img width='{int(display_px)}' height='{int(display_px)}' "
        f"src='data:image/gif;base64,{gif_b64}'/>"
        "</div>"
    )

    still_panel = ""
    if include_still:
        if still_hw is None:
            raise ValueError("still_hw is required when include_still=True")
        if still_camera_deg is None or still_light_deg is None:
            raise ValueError(
                "still_camera_deg and still_light_deg are required when "
                "include_still=True"
            )
        still_label = (
            f"{sequence_id}  cam={float(still_camera_deg):.0f}°  "
            f"light={float(still_light_deg):.0f}°"
        )
        png_bytes = build_anomaly_still_png(
            still_hw, label=still_label, mean=mean, std=std, display_px=display_px
        )
        png_b64 = b64encode(png_bytes).decode("ascii")
        still_cap = still_caption or (
            f"Still [1,1,{h},{w}]  cam={float(still_camera_deg):.0f}°  "
            f"light={float(still_light_deg):.0f}°"
        )
        still_panel = (
            "<div style='text-align:center;'>"
            f"{_panel_heading_html(still_heading)}"
            f"{_panel_caption_html(still_cap)}"
            f"<img width='{int(display_px)}' height='{int(display_px)}' "
            f"src='data:image/png;base64,{png_b64}'/>"
            "</div>"
        )

    return HTML(
        "<div style='display:flex;gap:1.25rem;align-items:flex-start;"
        "flex-wrap:wrap;'>"
        f"{still_panel}{gif_panel}"
        "</div>"
        "<p style='font: 12px sans-serif; max-width: 42rem;'>"
        "<b>Display only (fixed scale):</b> "
        f"mean={float(mean):g}, std={float(std):.0e}; "
        "±1 maps to black/white. The GIF walks the illumination-major grid "
        "<code>views[light_i, cam_j]</code> (all cameras per light, then next "
        "light), matching the canonical M10 sample tensor "
        "<code>[I, V, C, H, W]</code>. "
        "ML training uses per-image (per-view) z-score."
        "</p>"
    )


def display_anomaly_camera_light_grid(
    still_hw: np.ndarray | None,
    grid_iv: np.ndarray,
    *,
    sequence_id: str,
    camera_angles_deg: Sequence[float],
    light_angles_deg: Sequence[float],
    still_camera_deg: float | None = None,
    still_light_deg: float | None = None,
    mean: float = DEFAULT_DISPLAY_MEAN,
    std: float = DEFAULT_DISPLAY_STD,
    display_px: int = DEFAULT_DISPLAY_PX,
    fps: float = DEFAULT_CAMERA_LIGHT_GRID_FPS,
    include_still: bool = True,
    heading: str | None = None,
    caption: str | None = None,
    still_heading: str | None = None,
    still_caption: str | None = None,
) -> HTML:
    """Build and ``display`` an ``[I, V]`` grid GIF (optional still)."""
    html = anomaly_camera_light_grid_html(
        still_hw,
        grid_iv,
        sequence_id=sequence_id,
        still_camera_deg=still_camera_deg,
        still_light_deg=still_light_deg,
        camera_angles_deg=camera_angles_deg,
        light_angles_deg=light_angles_deg,
        mean=mean,
        std=std,
        display_px=display_px,
        fps=fps,
        include_still=include_still,
        heading=heading,
        caption=caption,
        still_heading=still_heading,
        still_caption=still_caption,
    )
    display(html)
    return html
