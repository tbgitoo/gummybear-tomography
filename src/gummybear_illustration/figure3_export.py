"""Export Figure 3 POV frames, renders, GIF, and final still from JSON history."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from gummybear.paths import display_path

from .export_m8_physical_scene import render_pov_file
from .figure3_history import (
    MODEL_FOURIER,
    MODEL_POOLING,
    build_figure3_record_index,
    default_figure3_root,
    history_paths,
    load_history,
    select_best_fourier_advantage_sample,
)
from .figure3_pov_scene import (
    DEFAULT_BALL_LIGHT_INTENSITY,
    DEFAULT_BALL_SPOTLIGHT_ANGLE_DEG,
    DEFAULT_BALL_SPOTLIGHTS,
    DEFAULT_CAMERA_DISTANCE_SCALE,
    DEFAULT_CAMERA_FOV_DEG,
    DEFAULT_CAMERA_YAW_DEG,
    DEFAULT_LIGHT_INTENSITY,
    FOURIER_LIGHT,
    FOURIER_RGB,
    POOLING_LIGHT,
    POOLING_RGB,
    figure3_apply_bear_placement,
    figure3_bear_object_pov_text,
    figure3_camera_pose,
    figure3_markers_pov_text,
    figure3_mesh_metrics,
    _emission_rgb,
    _marker_ball,
    figure3_world_pov_text,
    normalize_figure3_split,
    prepare_bear_inc,
    tracked_sample_ids,
)
from .figure3_train import train_figure3_convergence
from .network_captions import COLOR_FOURIER, COLOR_GAP, COLOR_TARGET
from .paths import default_cad_dir, repo_root
from .pov_primitives import pov_vec


def figure3_layout(root: Path) -> dict[str, Path]:
    root = Path(root)
    hist = history_paths(root)
    return {
        **hist,
        "figure_root": root,
        "povray": root / "povray",
        "scenes": root / "povray" / "scenes",
        "renders": root / "povray" / "renders",
        "final": root / "final",
        "final_png": root / "final" / "figure3_m8_vs_fourier_convergence.png",
        "final_gif": root / "final" / "figure3_m8_vs_fourier_convergence.gif",
        "summary_json": root / "final" / "figure3_summary.json",
    }


def default_pov_render_workers() -> int:
    """Conservative local worker count: available CPUs minus two, at least 1."""
    return max(1, (os.cpu_count() or 1) - 2)


def figure3_render_worker_count(
    n_frames: int,
    *,
    multiprocessing: bool = True,
) -> int:
    """How many concurrent POV-Ray processes to launch."""
    n = max(int(n_frames), 0)
    if not multiprocessing or n <= 1:
        return 1
    return max(1, min(n, default_pov_render_workers()))


def _figure3_process_pool_context() -> mp.context.BaseContext:
    """Process pool context for Figure 3 workers.

    Prefer ``fork`` on Unix so Jupyter/IPython kernels and large worker state
    (record index, mesh) work reliably. Fall back to ``spawn`` elsewhere.
    """
    if sys.platform != "win32":
        for method in ("fork", "spawn"):
            try:
                return mp.get_context(method)
            except ValueError:
                continue
    return mp.get_context("spawn")


def _render_one_pov_png(
    pov_path: str,
    png_path: str,
    width: int,
    height: int,
    work_threads: int | None,
) -> str | None:
    """Picklable worker: render one ``.pov`` to PNG. Return the PNG path or None."""
    out = render_pov_file(
        Path(pov_path),
        Path(png_path),
        width=width,
        height=height,
        work_threads=work_threads,
    )
    return None if out is None else str(out)


_MARKER_INC_WORKER: dict[str, Any] = {}


def _init_marker_inc_worker(state: dict[str, Any]) -> None:
    global _MARKER_INC_WORKER
    _MARKER_INC_WORKER = state


def _generate_marker_inc_for_epoch(epoch: int) -> tuple[str, str]:
    """Picklable worker: build one marker ``.inc`` filename and body."""
    state = _MARKER_INC_WORKER
    text = figure3_markers_pov_text(
        combined_history=None,
        epoch=int(epoch),
        mesh=state["mesh"],
        loc=state["loc"],
        split=state.get("split"),
        bear_id=state["bear_id"],
        bear_scale=state["bear_scale"],
        bear_translate=state.get("bear_translate"),
        record_index=state["record_index"],
        split_sample_ids=state["split_sample_ids"],
        **state.get("marker_kw", {}),
    )
    name = state["template"].format(epoch=int(epoch))
    return name, text


def _write_marker_includes_sequential(
    *,
    scenes_dir: Path,
    epochs: Sequence[int],
    template: str,
    mesh: trimesh.Trimesh,
    loc: np.ndarray,
    split: str,
    bear_id: str,
    bear_scale: float,
    bear_translate: Sequence[float] | None,
    record_index: Any,
    split_sample_ids: Sequence[str],
    marker_kw: Mapping[str, Any],
) -> None:
    for epoch in epochs:
        text = figure3_markers_pov_text(
            combined_history=None,
            epoch=int(epoch),
            mesh=mesh,
            loc=loc,
            split=split,
            bear_id=bear_id,
            bear_scale=bear_scale,
            bear_translate=bear_translate,
            record_index=record_index,
            split_sample_ids=split_sample_ids,
            **marker_kw,
        )
        name = template.format(epoch=int(epoch))
        (scenes_dir / name).write_text(text, encoding="utf-8")


def _write_marker_includes_parallel(
    *,
    scenes_dir: Path,
    epochs: Sequence[int],
    template: str,
    mesh: trimesh.Trimesh,
    loc: np.ndarray,
    split: str,
    bear_id: str,
    bear_scale: float,
    bear_translate: Sequence[float] | None,
    record_index: Any,
    split_sample_ids: Sequence[str],
    marker_kw: Mapping[str, Any],
    multiprocessing: bool = True,
) -> None:
    seq_kw = {
        "scenes_dir": scenes_dir,
        "epochs": epochs,
        "template": template,
        "mesh": mesh,
        "loc": loc,
        "split": split,
        "bear_id": bear_id,
        "bear_scale": bear_scale,
        "bear_translate": bear_translate,
        "record_index": record_index,
        "split_sample_ids": split_sample_ids,
        "marker_kw": marker_kw,
    }
    n = len(epochs)
    workers = figure3_render_worker_count(n, multiprocessing=multiprocessing)
    if workers <= 1:
        _write_marker_includes_sequential(**seq_kw)
        return

    worker_state = {
        "template": template,
        "mesh": mesh,
        "loc": loc,
        "split": split,
        "bear_id": bear_id,
        "bear_scale": bear_scale,
        "bear_translate": bear_translate,
        "record_index": record_index,
        "split_sample_ids": list(split_sample_ids),
        "marker_kw": dict(marker_kw),
    }
    mp_ctx = _figure3_process_pool_context()
    print(
        f"Writing {n} marker includes with {workers} processes "
        f"(start method={mp_ctx.get_start_method()}).",
        flush=True,
    )
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_ctx,
            initializer=_init_marker_inc_worker,
            initargs=(worker_state,),
        ) as pool:
            for name, text in pool.map(_generate_marker_inc_for_epoch, epochs):
                (scenes_dir / name).write_text(text, encoding="utf-8")
    except Exception as exc:
        print(
            "Parallel marker include generation failed "
            f"({type(exc).__name__}: {exc}); falling back to sequential.",
            flush=True,
        )
        _write_marker_includes_sequential(**seq_kw)


def _resolve_stl(
    *,
    repo: Path,
    combined: Mapping[str, Any],
    sample_id: str,
    stl_path: Path | None,
) -> Path:
    if stl_path is not None and Path(stl_path).is_file():
        return Path(stl_path)
    for meta in combined.get("tracked_samples") or []:
        if str(meta.get("sample_id")) != str(sample_id):
            continue
        seq_dir = meta.get("sequence_dir")
        if seq_dir:
            cand = Path(seq_dir) / "geometry" / "phantom.stl"
            if cand.is_file():
                return cand
            # common M8 layout: sequence_dir itself may contain stl link via manifest
        man = meta.get("manifest_path")
        if man and Path(man).is_file():
            blob = json.loads(Path(man).read_text(encoding="utf-8"))
            phantom = blob.get("phantom") if isinstance(blob.get("phantom"), dict) else {}
            stl = (
                blob.get("stl_path")
                or phantom.get("stl_path")
                or blob.get("geometry", {}).get("stl_path")
            )
            if stl:
                p = Path(stl)
                if not p.is_absolute():
                    p = repo / p
                if p.is_file():
                    return p
    for name in ("proto_bear_head.stl", "proto_bear.stl"):
        proto = default_cad_dir(repo) / name
        if proto.is_file():
            return proto
    raise FileNotFoundError(
        "Could not resolve an STL for Figure 3 (tracked sample manifest / "
        f"cad/*.stl). sample_id={sample_id!r}"
    )


def _sans(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size=size)


def _figure3_epoch0_train_loss_from_preds(
    combined: Mapping[str, Any],
    *,
    model_type: str,
) -> float | None:
    """Mean per-sample MSE of ``y_pred`` vs ``y_true`` at epoch 0 (init).

    Used when histories predate logging ``train_loss`` at initialization.
    """
    mses: list[float] = []
    for rec in combined.get("records") or []:
        if str(rec.get("model_type")) != str(model_type):
            continue
        if int(rec.get("epoch", -1)) != 0:
            continue
        yt = rec.get("y_true")
        yp = rec.get("y_pred")
        if yt is None or yp is None:
            continue
        yt_a = np.asarray(yt, dtype=float).reshape(-1)
        yp_a = np.asarray(yp, dtype=float).reshape(-1)
        if yt_a.size == 0 or yt_a.shape != yp_a.shape:
            continue
        mses.append(float(np.mean((yp_a - yt_a) ** 2)))
    if not mses:
        return None
    return float(np.mean(mses))


def _figure3_train_loss_by_epoch(
    combined: Mapping[str, Any],
    *,
    model_type: str,
) -> list[tuple[int, float]]:
    """Unique ``(epoch, train_loss)`` points for one model, sorted by epoch.

    If epoch 0 has predictions but no logged ``train_loss`` (older histories),
    synthesize the init point from ``y_pred`` / ``y_true`` MSE.
    """
    by_epoch: dict[int, float] = {}
    for rec in combined.get("records") or []:
        if str(rec.get("model_type")) != str(model_type):
            continue
        loss = rec.get("train_loss")
        if loss is None:
            continue
        epoch = int(rec.get("epoch", 0))
        if epoch not in by_epoch:
            by_epoch[epoch] = float(loss)
    if 0 not in by_epoch:
        init_loss = _figure3_epoch0_train_loss_from_preds(
            combined, model_type=model_type
        )
        if init_loss is not None and init_loss > 0.0:
            by_epoch[0] = init_loss
    return sorted(by_epoch.items())


def _loss_at_or_before_epoch(
    series: Sequence[tuple[int, float]],
    epoch: int,
) -> tuple[int, float] | None:
    out: tuple[int, float] | None = None
    for e, loss in series:
        if int(e) > int(epoch):
            break
        out = (int(e), float(loss))
    return out


def _draw_glow_dot(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    *,
    color: tuple[int, int, int],
    radius: float,
) -> None:
    cx, cy = center
    for scale, alpha in ((3.8, 34), (2.6, 56), (1.6, 88)):
        rr = radius * scale
        fill = (*color, alpha)
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=fill)
    rr = radius
    draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(*color, 255))
    draw.ellipse(
        (cx - 0.42 * rr, cy - 0.42 * rr, cx + 0.42 * rr, cy + 0.42 * rr),
        fill=(245, 245, 255, 235),
    )



FIGURE3_BOARD_TEXTURE_WIDTH = 3200
FIGURE3_BOARD_TEXTURE_HEIGHT = 2200
FIGURE3_BOARD_TITLE = "Training loss"


def _figure3_board_axis_labels(combined: Mapping[str, Any]) -> tuple[str, str, str, str, float, float, int] | None:
    """Return ``(y_top, y_bot, x_left, x_right, y_min, y_max, x_den)`` or None."""
    pooled = _figure3_train_loss_by_epoch(combined, model_type=MODEL_POOLING)
    fourier = _figure3_train_loss_by_epoch(combined, model_type=MODEL_FOURIER)
    positive = [
        loss
        for series in (pooled, fourier)
        for _e, loss in series
        if loss > 0.0
    ]
    if not positive:
        return None
    y_min = min(positive)
    y_max = max(positive)
    y_min = 10 ** np.floor(np.log10(max(y_min * 0.8, 1e-9)))
    y_max = 10 ** np.ceil(np.log10(max(y_max * 1.25, y_min * 1.2)))
    x_den = max(
        int(combined.get("num_epochs", 0)),
        max((e for series in (pooled, fourier) for e, _l in series), default=1),
        1,
    )

    def _sci(val: float) -> str:
        decade = int(round(np.log10(max(float(val), 1e-30))))
        return f"1e{decade}"

    return _sci(y_max), _sci(y_min), "0", str(x_den), float(y_min), float(y_max), int(x_den)


def _figure3_board_plot_pads(
    *,
    width: int,
    height: int,
    y_top: str,
    y_bot: str,
    x_left: str,
    x_right: str,
    label_font_px: int | None = None,
    title: str = FIGURE3_BOARD_TITLE,
) -> tuple[float, float, float, float, Any, Any]:
    """Pad fractions and fonts sized so title + end-tick labels fit in the PNG.

    ``label_font_px`` is the preferred FreeType pixel size for axis ticks. If
    glyphs would not fit, the size is stepped down until they do. The title
    font is slightly larger than the tick font.
    """
    w = max(int(width), 16)
    h = max(int(height), 16)
    probe = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(probe)
    edge = max(8, int(0.02 * min(w, h)))
    if label_font_px is None:
        target = max(96, int(0.12 * h))
    else:
        target = max(24, int(label_font_px))
    font = _sans(target)
    title_font = font
    title_text = str(title or "").strip()

    def _size(msg: str, fnt: Any) -> tuple[int, int]:
        box = draw.textbbox((0, 0), msg, font=fnt, anchor="lt")
        return int(box[2] - box[0]), int(box[3] - box[1])

    candidates = [target]
    for frac in (0.85, 0.70, 0.55, 0.40, 0.28):
        candidates.append(max(24, int(target * frac)))
    candidates.append(48)

    for size in candidates:
        font = _sans(size)
        title_font = _sans(max(size, int(round(1.2 * size))))
        y_tw = max(_size(y_top, font)[0], _size(y_bot, font)[0])
        y_th = max(_size(y_top, font)[1], _size(y_bot, font)[1])
        x_tw = max(_size(x_left, font)[0], _size(x_right, font)[0])
        x_th = max(_size(x_left, font)[1], _size(x_right, font)[1])
        title_h = _size(title_text, title_font)[1] if title_text else 0
        title_w = _size(title_text, title_font)[0] if title_text else 0
        xpad_l = (y_tw + 2 * edge) / float(w)
        xpad_r = (x_tw // 2 + edge) / float(w)
        # Top pad: title band + gap + half of the top y-tick (anchored at py0).
        ypad_t = (edge + title_h + edge + y_th // 2) / float(h) if title_text else (
            (y_th // 2 + edge) / float(h)
        )
        ypad_b = (x_th + 2 * edge) / float(h)
        title_fits = (not title_text) or (title_w < 0.92 * w)
        if (
            xpad_l + xpad_r < 0.72
            and ypad_t + ypad_b < 0.78
            and title_fits
        ):
            return (
                float(xpad_l),
                float(xpad_r),
                float(ypad_t),
                float(ypad_b),
                font,
                title_font,
            )
    return 0.40, 0.14, 0.28, 0.40, font, title_font


def _draw_figure3_loss_board_image(
    *,
    width: int,
    height: int,
    combined: Mapping[str, Any],
    label_font_px: int | None = None,
    curve_width_px: int | None = None,
) -> Image.Image:
    """Chalkboard texture: title, thick curves, only end-tick labels."""
    w = max(int(width), 16)
    h = max(int(height), 16)
    image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    board = (0, 0, w - 1, h - 1)
    x0, y0, x1, y1 = board
    draw.rounded_rectangle(
        board,
        radius=max(12, int(0.012 * min(w, h))),
        fill=(18, 20, 18, 255),
        outline=(120, 120, 115, 220),
        width=max(3, int(0.004 * w)),
    )
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)

    labels = _figure3_board_axis_labels(combined)
    if labels is None:
        return image
    y_top, y_bot, x_left, x_right, y_min, y_max, x_den = labels
    xpad_l, xpad_r, ypad_t, ypad_b, label_font, title_font = _figure3_board_plot_pads(
        width=w,
        height=h,
        y_top=y_top,
        y_bot=y_bot,
        x_left=x_left,
        x_right=x_right,
        label_font_px=label_font_px,
    )
    px0 = x0 + int(xpad_l * bw)
    px1 = x1 - int(xpad_r * bw)
    py0 = y0 + int(ypad_t * bh)
    py1 = y1 - int(ypad_b * bh)
    plot_w = max(px1 - px0, 1)
    plot_h = max(py1 - py0, 1)
    log_lo = float(np.log10(y_min))
    log_hi = float(np.log10(y_max))
    log_span = max(log_hi - log_lo, 1e-6)

    def map_xy(ep: int, loss: float) -> tuple[float, float]:
        xf = (float(ep) / float(x_den)) if x_den > 0 else 0.0
        yf = (np.log10(max(float(loss), y_min)) - log_lo) / log_span
        return px0 + xf * plot_w, py1 - float(yf) * plot_h

    axis_w = max(8, int(0.01 * w))
    if curve_width_px is None:
        curve_w = max(30, int(0.054 * w))
    else:
        curve_w = max(1, int(curve_width_px))
    ink = (245, 245, 235, 255)
    edge = max(8, int(0.02 * min(w, h)))

    draw.text(
        ((x0 + x1) / 2.0, py0 - edge),
        FIGURE3_BOARD_TITLE,
        fill=ink,
        font=title_font,
        anchor="mb",
    )
    draw.line((px0, py0, px0, py1), fill=ink, width=axis_w)
    draw.line((px0, py1, px1, py1), fill=ink, width=axis_w)
    draw.text((px0 - edge, py0), y_top, fill=ink, font=label_font, anchor="rm")
    draw.text((px0 - edge, py1), y_bot, fill=ink, font=label_font, anchor="rm")
    # ``mt``: anchor at top of glyphs so the whole digit sits in the bottom pad.
    draw.text((px0, py1 + edge), x_left, fill=ink, font=label_font, anchor="mt")
    draw.text((px1, py1 + edge), x_right, fill=ink, font=label_font, anchor="mt")

    pooled = _figure3_train_loss_by_epoch(combined, model_type=MODEL_POOLING)
    fourier = _figure3_train_loss_by_epoch(combined, model_type=MODEL_FOURIER)
    styles = {
        MODEL_POOLING: (235, 55, 55, 255),
        MODEL_FOURIER: (70, 130, 255, 255),
    }
    for model_type, series in (
        (MODEL_POOLING, pooled),
        (MODEL_FOURIER, fourier),
    ):
        if len(series) < 2:
            continue
        pts = [map_xy(ep, loss) for ep, loss in series]
        draw.line(pts, fill=styles[model_type], width=curve_w, joint="curve")

    return image


def _write_figure3_loss_board_png(
    png_path: Path,
    *,
    combined: Mapping[str, Any],
    width: int = FIGURE3_BOARD_TEXTURE_WIDTH,
    height: int = FIGURE3_BOARD_TEXTURE_HEIGHT,
    label_font_px: int | None = None,
    curve_width_px: int | None = None,
) -> Path:
    image = _draw_figure3_loss_board_image(
        width=width,
        height=height,
        combined=combined,
        label_font_px=label_font_px,
        curve_width_px=curve_width_px,
    )
    # POV-Ray treats PNG alpha as filter/transmit; flatten to opaque RGB so the
    # chalk curves stay fully visible on the board face.
    if image.mode == "RGBA":
        flat = Image.new("RGB", image.size, (14, 16, 14))
        flat.paste(image, mask=image.split()[3])
        image = flat
    else:
        image = image.convert("RGB")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)
    return png_path


def _camera_basis(
    loc: np.ndarray,
    look: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal camera frame: forward (look), right, up (z-up world)."""
    loc_a = np.asarray(loc, dtype=float).reshape(3)
    look_a = np.asarray(look, dtype=float).reshape(3)
    up_ref = np.array([0.0, 0.0, 1.0], dtype=float)
    fwd = look_a - loc_a
    fn = float(np.linalg.norm(fwd))
    if fn < 1e-9:
        fwd = np.array([0.0, 1.0, 0.0], dtype=float)
        fn = 1.0
    fwd = fwd / fn
    # ``cross(up, fwd)`` is the viewer's right; ``cross(fwd, up)`` points left.
    right = np.cross(up_ref, fwd)
    rn = float(np.linalg.norm(right))
    if rn < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
        rn = 1.0
    right = right / rn
    up = np.cross(fwd, right)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    return fwd, right, up


def _figure3_loss_board_pov_text(
    *,
    png_name: str,
    loc: np.ndarray,
    look: np.ndarray,
    board_center: np.ndarray,
    half_width: float,
    half_height: float,
    thickness: float,
) -> str:
    """Chalkboard body + face texture.

    Both the dark body and the curve texture are unit boxes placed with the
    same camera-basis ``matrix``. A POV ``box { A, B }`` is axis-aligned, so
    using world-space corners for the body previously filled an AABB that sat
    in front of the lower (farther) part of the oriented face and hid it.
    """
    fwd, right, up = _camera_basis(loc, look)
    center = np.asarray(board_center, dtype=float).reshape(3)
    hw = float(half_width)
    hh = float(half_height)
    thick = max(float(thickness), 1e-3)
    # Camera looks along +fwd; camera-facing side of the body is at -thick*fwd
    # from ``center`` (body occupies local Z in [-1, 0] after the matrix below).
    origin = center - hw * right - hh * up - thick * fwd
    ax = 2.0 * hw * right
    ay = 2.0 * hh * up
    az = thick * fwd
    face_eps = max(0.06 * thick, 0.04)
    face_az = face_eps * fwd
    face_origin = origin - 0.5 * face_eps * fwd

    def _matrix(ax_: np.ndarray, ay_: np.ndarray, az_: np.ndarray, origin_: np.ndarray) -> str:
        return (
            "    matrix <\n"
            f"      {ax_[0]:.6g}, {ax_[1]:.6g}, {ax_[2]:.6g},\n"
            f"      {ay_[0]:.6g}, {ay_[1]:.6g}, {ay_[2]:.6g},\n"
            f"      {az_[0]:.6g}, {az_[1]:.6g}, {az_[2]:.6g},\n"
            f"      {origin_[0]:.6g}, {origin_[1]:.6g}, {origin_[2]:.6g}\n"
            "    >\n"
        )

    bits = [
        "// Figure 3 learning-rate chalkboard.\n",
        "union {\n",
        (
            "  box {\n"
            "    <0, 0, 0>, <1, 1, 1>\n"
            "    pigment { color rgb <0.05, 0.055, 0.05> }\n"
            "    finish { ambient 0.03 diffuse 0.78 specular 0.05 roughness 0.08 }\n"
            f"{_matrix(ax, ay, az, origin)}"
            "  }\n"
        ),
        (
            "  box {\n"
            "    <0, 0, -0.5>, <1, 1, 0.5>\n"
            "    texture {\n"
            "      pigment {\n"
            f'        image_map {{ png "{png_name}" once interpolate 2 }}\n'
            "      }\n"
            "      finish { ambient 0.7 diffuse 0.45 specular 0.04 }\n"
            "    }\n"
            "    double_illuminate\n"
            f"{_matrix(ax, ay, face_az, face_origin)}"
            "  }\n"
        ),
        "}\n",
    ]
    return "".join(bits)


def _figure3_loss_board_dot_pov_text(
    *,
    epoch: int,
    num_epochs: int,
    combined: Mapping[str, Any],
    loc: np.ndarray,
    look: np.ndarray,
    board_center: np.ndarray,
    half_width: float,
    half_height: float,
    thickness: float,
    label_font_px: int | None = None,
    dot_radius_mm: float | None = None,
) -> str:
    """Per-epoch luminous dots on the board plane."""
    pooled = _figure3_train_loss_by_epoch(combined, model_type=MODEL_POOLING)
    fourier = _figure3_train_loss_by_epoch(combined, model_type=MODEL_FOURIER)
    positive = [
        loss
        for series in (pooled, fourier)
        for _ep, loss in series
        if float(loss) > 0.0
    ]
    if not positive:
        return "// Figure 3 learning-rate board dots omitted: no train_loss.\n"
    fwd, right, up = _camera_basis(loc, look)
    center = np.asarray(board_center, dtype=float).reshape(3)
    board_w = 2.0 * float(half_width)
    board_h = 2.0 * float(half_height)
    labels = _figure3_board_axis_labels(combined)
    if labels is None:
        return "// Figure 3 learning-rate board dots omitted: no train_loss.\n"
    y_top, y_bot, x_left, x_right, _y_min, _y_max, _x_den = labels
    xpad_l_f, xpad_r_f, ypad_t_f, ypad_b_f, _font, _title_font = _figure3_board_plot_pads(
        width=FIGURE3_BOARD_TEXTURE_WIDTH,
        height=FIGURE3_BOARD_TEXTURE_HEIGHT,
        y_top=y_top,
        y_bot=y_bot,
        x_left=x_left,
        x_right=x_right,
        label_font_px=label_font_px,
    )
    xpad_l = xpad_l_f * board_w
    xpad_r = xpad_r_f * board_w
    ypad_t = ypad_t_f * board_h
    ypad_b = ypad_b_f * board_h
    plot_w = max(board_w - xpad_l - xpad_r, 1e-9)
    plot_h = max(board_h - ypad_t - ypad_b, 1e-9)
    y_min = min(positive)
    y_max = max(positive)
    y_min = 10 ** np.floor(np.log10(max(y_min * 0.8, 1e-9)))
    y_max = 10 ** np.ceil(np.log10(max(y_max * 1.25, y_min * 1.2)))
    log_lo = float(np.log10(y_min))
    log_hi = float(np.log10(y_max))
    log_span = max(log_hi - log_lo, 1e-6)
    x_den = max(
        int(num_epochs),
        max((e for series in (pooled, fourier) for e, _loss in series), default=1),
        1,
    )

    def board_xy(ep: int, loss: float) -> tuple[float, float]:
        xf = (float(ep) / float(x_den)) if x_den > 0 else 0.0
        yf = (np.log10(max(float(loss), y_min)) - log_lo) / log_span
        x_local = -float(half_width) + xpad_l + xf * plot_w
        z_local = -float(half_height) + ypad_b + yf * plot_h
        return x_local, z_local

    # Sit clearly in front of the camera-facing texture (body ends at -thickness*fwd).
    plane_origin = center - (float(thickness) + max(0.25, 0.008 * board_w)) * fwd
    if dot_radius_mm is None:
        # ~2.5% of the shorter board edge — a discrete bead on the curve.
        dot_r = max(0.55, 0.025 * min(board_w, board_h))
    else:
        dot_r = max(0.2, float(dot_radius_mm))
    glow_r = 1.7 * dot_r
    bits = ["// Figure 3 learning-rate board dots.\n"]
    for series, rgb, light_rgb in (
        (pooled, POOLING_RGB, POOLING_LIGHT),
        (fourier, FOURIER_RGB, FOURIER_LIGHT),
    ):
        spot = _loss_at_or_before_epoch(series, epoch)
        if spot is None:
            continue
        ep_now, loss_now = spot
        x_local, z_local = board_xy(ep_now, loss_now)
        c = plane_origin + x_local * right + z_local * up
        # Soft halo only (no light_source) — readable on the chalk, not a flood.
        bits.append(
            _marker_ball(
                c,
                radius=glow_r,
                rgb=rgb,
                light_rgb=light_rgb,
                intensity=0.0,
                emission=_emission_rgb(light_rgb, 0.18),
                fade_distance=2.0,
                emit_light=False,
                transmit=0.82,
                hollow=True,
            )
        )
        bits.append(
            _marker_ball(
                c,
                radius=dot_r,
                rgb=rgb,
                light_rgb=light_rgb,
                intensity=0.22,
                emission=_emission_rgb(light_rgb, 0.48),
                fade_distance=3.0,
                emit_light=True,
                spotlight=False,
                fade_power=2.0,
            )
        )
    return "".join(bits)


FIGURE3_WORLD_INC = "figure3_world.inc"
FIGURE3_BEAR_MESH_INC = "figure3_bear.inc"
FIGURE3_BEAR_OBJECT_INC = "figure3_bear_object.inc"
FIGURE3_DEFAULT_BEAR_ID = "default"
FIGURE3_BOARD_TEXTURE_DIR = "figure3_loss_board_pngs"
FIGURE3_BOARD_TEXTURE_NAME = "figure3_loss_board_curves.png"
FIGURE3_BOARD_OBJECT_INC = "figure3_board_object.inc"
_INCLUDE_RE = re.compile(r'^[ \t]*#include\s+"([^"]+)"', re.MULTILINE)


def _layer_token(name: str) -> str:
    raw = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name).strip()
    )
    return raw.strip("_") or FIGURE3_DEFAULT_BEAR_ID


def _bear_inc_names(bear_id: str) -> tuple[str, str]:
    token = _layer_token(bear_id)
    if token == FIGURE3_DEFAULT_BEAR_ID:
        return FIGURE3_BEAR_MESH_INC, FIGURE3_BEAR_OBJECT_INC
    return f"figure3_bear_{token}.inc", f"figure3_bear_{token}_object.inc"


def _marker_inc_template(*, bear_id: str, split: str) -> str:
    return (
        "figure3_epoch_{epoch:04d}_markers_"
        f"{_layer_token(bear_id)}_{normalize_figure3_split(split)}.inc"
    )


def _board_inc_template(*, board_id: str = "loss") -> str:
    return f"figure3_epoch_{{epoch:04d}}_board_{_layer_token(board_id)}.inc"


@dataclass
class Figure3SceneSet:
    """On-disk progressive POV-Ray scene set (world, then bear, then markers)."""

    layout: dict[str, Path]
    repo: Path
    mesh: Any
    stl_path: Path
    camera: dict[str, list[float]]
    epochs: list[int]
    pov_paths: list[Path]
    loc: np.ndarray
    look: np.ndarray
    includes: list[str] = field(default_factory=list)
    bears: dict[str, dict[str, Any]] = field(default_factory=dict)
    marker_layers: list[dict[str, str]] = field(default_factory=list)
    board_layers: list[dict[str, str]] = field(default_factory=list)
    record_index: Any = None
    fov_deg: float = DEFAULT_CAMERA_FOV_DEG
    yaw_deg: float = DEFAULT_CAMERA_YAW_DEG
    light_intensity: float = DEFAULT_LIGHT_INTENSITY
    distance_scale: float = DEFAULT_CAMERA_DISTANCE_SCALE

    @property
    def scenes_dir(self) -> Path:
        return Path(self.layout["scenes"])

    @property
    def marker_template(self) -> str | None:
        """Most recently added marker include template (back-compat)."""
        if not self.marker_layers:
            return None
        return self.marker_layers[-1]["template"]


def composed_figure3_pov_text(pov_path: Path) -> str:
    """Inline local ``#include`` files (for tests / inspection)."""
    pov_path = Path(pov_path)
    text = pov_path.read_text(encoding="utf-8")

    def repl(match: re.Match[str]) -> str:
        inc = pov_path.parent / match.group(1)
        if not inc.is_file():
            return match.group(0)
        return composed_figure3_pov_text(inc)

    return _INCLUDE_RE.sub(repl, text)


def figure3_epoch_list(
    combined: Mapping[str, Any],
    *,
    num_epochs: int | None = None,
    stride: int = 1,
) -> list[int]:
    num_ep = int(combined.get("num_epochs", num_epochs or 0))
    step = max(int(stride), 1)
    epochs = list(range(0, num_ep + 1, step))
    if not epochs:
        epochs = [0]
    if epochs[-1] != num_ep:
        epochs.append(num_ep)
    return epochs


def _rewrite_epoch_povs(scenes: Figure3SceneSet) -> None:
    scenes.scenes_dir.mkdir(parents=True, exist_ok=True)
    for epoch, pov in zip(scenes.epochs, scenes.pov_paths, strict=True):
        bits = ["#version 3.7;\n"]
        for inc in scenes.includes:
            bits.append(f'#include "{inc}"\n')
        for layer in scenes.marker_layers:
            bits.append(
                f'#include "{layer["template"].format(epoch=int(epoch))}"\n'
            )
        for layer in scenes.board_layers:
            bits.append(
                f'#include "{layer["template"].format(epoch=int(epoch))}"\n'
            )
        Path(pov).write_text("".join(bits), encoding="utf-8")


def _append_include(scenes: Figure3SceneSet, name: str) -> None:
    if name not in scenes.includes:
        scenes.includes.append(name)


def generate_figure3_scenes(
    *,
    layout: Mapping[str, Path],
    repo: Path,
    epochs: Sequence[int],
    combined: Mapping[str, Any] | None = None,
    stl_path: Path | None = None,
    sample_id: str | None = None,
    camera_location: Sequence[float] | None = None,
    camera_look_at: Sequence[float] | None = None,
    fov_deg: float | None = None,
    distance_scale: float | None = None,
    yaw_deg: float | None = None,
    light_intensity: float | None = None,
) -> Figure3SceneSet:
    """Write camera/lights/floor world include and stub per-epoch ``.pov`` files."""
    layout_d = dict(layout)
    scenes_dir = Path(layout_d["scenes"])
    scenes_dir.mkdir(parents=True, exist_ok=True)
    combined = {} if combined is None else combined
    sid = sample_id
    if sid is None:
        tracked = combined.get("tracked_samples") or []
        sid = str(tracked[0]["sample_id"]) if tracked else ""
    stl = _resolve_stl(
        repo=Path(repo),
        combined=combined,
        sample_id=str(sid),
        stl_path=stl_path,
    )
    mesh = trimesh.load(stl, force="mesh")
    pose_kw: dict[str, Any] = {}
    if distance_scale is not None:
        pose_kw["distance_scale"] = float(distance_scale)
    if yaw_deg is not None:
        pose_kw["yaw_deg"] = float(yaw_deg)
    loc, look = figure3_camera_pose(
        mesh,
        camera_location=camera_location,
        camera_look_at=camera_look_at,
        **pose_kw,
    )
    fov = DEFAULT_CAMERA_FOV_DEG if fov_deg is None else float(fov_deg)
    yaw = DEFAULT_CAMERA_YAW_DEG if yaw_deg is None else float(yaw_deg)
    light = DEFAULT_LIGHT_INTENSITY if light_intensity is None else float(light_intensity)
    dist = (
        DEFAULT_CAMERA_DISTANCE_SCALE
        if distance_scale is None
        else float(distance_scale)
    )
    world = figure3_world_pov_text(
        mesh=mesh,
        loc=loc,
        look=look,
        fov_deg=fov,
        yaw_deg=yaw,
        light_intensity=light,
        with_version=False,
    )
    (scenes_dir / FIGURE3_WORLD_INC).write_text(world, encoding="utf-8")
    epoch_list = [int(e) for e in epochs]
    pov_paths = [
        scenes_dir / f"figure3_epoch_{int(epoch):04d}.pov" for epoch in epoch_list
    ]
    scenes = Figure3SceneSet(
        layout=layout_d,
        repo=Path(repo),
        mesh=mesh,
        stl_path=stl,
        camera={
            "location": [float(v) for v in loc],
            "look_at": [float(v) for v in look],
        },
        epochs=epoch_list,
        pov_paths=pov_paths,
        loc=loc,
        look=look,
        includes=[FIGURE3_WORLD_INC],
        fov_deg=fov,
        yaw_deg=yaw,
        light_intensity=light,
        distance_scale=dist,
    )
    _rewrite_epoch_povs(scenes)
    return scenes


def add_bear_to_figure3_scenes(
    scenes: Figure3SceneSet,
    *,
    bear_id: str = FIGURE3_DEFAULT_BEAR_ID,
    stl_path: Path | None = None,
    scale: float = 1.0,
    translate: Sequence[float] | None = None,
) -> Figure3SceneSet:
    """Add a named gummybear mesh instance (written once, included by every epoch).

    Call again with a different ``bear_id`` to place additional bears. Marker
    layers later target a bear with the same id.

    ``scale`` and ``translate`` scale about the mesh centroid then shift in
    simulation millimetres (negative x moves left in the default view).
    """
    token = _layer_token(bear_id)
    stl = Path(stl_path) if stl_path is not None else Path(scenes.stl_path)
    mesh_inc, object_inc = _bear_inc_names(token)
    bear_inc, mesh = prepare_bear_inc(stl, scenes.scenes_dir / mesh_inc)
    placement_translate = (
        None
        if translate is None
        else tuple(float(v) for v in np.asarray(translate, dtype=float).reshape(3))
    )
    object_text = figure3_bear_object_pov_text(
        mesh=mesh,
        bear_inc_name=bear_inc,
        include_mesh=False,
        scale=float(scale),
        translate=placement_translate,
    )
    (scenes.scenes_dir / object_inc).write_text(object_text, encoding="utf-8")
    scenes.bears[token] = {
        "bear_id": token,
        "mesh": mesh,
        "stl_path": stl,
        "mesh_inc": mesh_inc,
        "object_inc": object_inc,
        "scale": float(scale),
        "translate": placement_translate,
    }
    scenes.mesh = mesh
    scenes.stl_path = stl
    _append_include(scenes, mesh_inc)
    _append_include(scenes, object_inc)
    _rewrite_epoch_povs(scenes)
    return scenes


def add_markers_to_figure3_scenes(
    scenes: Figure3SceneSet,
    combined: Mapping[str, Any],
    *,
    split: str = "validation",
    bear_id: str = FIGURE3_DEFAULT_BEAR_ID,
    multiprocessing: bool = True,
    **marker_kw: Any,
) -> Figure3SceneSet:
    """Add one marker layer: ``split`` samples attached to ``bear_id``.

    ``split`` is ``train``, ``validation``, or ``test`` (aliases: training, val).
    Each (bear, split) pair writes its own per-epoch include so you can add
    several splits and several bears without overwriting earlier layers.
    Histories from ``train_figure3_convergence`` log train, validation, and test.

    Marker includes are generated in parallel across epochs when
    ``multiprocessing`` is true (same worker policy as ``render_figure3_frames``).
    """
    marker_kw = {k: v for k, v in marker_kw.items() if v is not None}
    split_n = normalize_figure3_split(split)
    token = _layer_token(bear_id)
    if token != FIGURE3_DEFAULT_BEAR_ID and token not in scenes.bears:
        known = sorted(scenes.bears) or [FIGURE3_DEFAULT_BEAR_ID]
        raise ValueError(
            f"bear_id={bear_id!r} has not been added; known bears: {known}. "
            "Call add_bear_to_figure3_scenes(..., bear_id=...) first."
        )
    if token in scenes.bears:
        mesh = scenes.bears[token]["mesh"]
        bear_scale = float(scenes.bears[token].get("scale", 1.0))
        bear_translate = scenes.bears[token].get("translate")
    elif scenes.mesh is not None:
        mesh = scenes.mesh
        bear_scale = 1.0
        bear_translate = None
    else:
        raise ValueError("no bear mesh on the scene set; add a bear first")
    if scenes.record_index is None:
        scenes.record_index = build_figure3_record_index(combined)
    split_sample_ids = tracked_sample_ids(combined, split=split_n)
    template = _marker_inc_template(bear_id=token, split=split_n)
    _write_marker_includes_parallel(
        scenes_dir=scenes.scenes_dir,
        epochs=scenes.epochs,
        template=template,
        mesh=mesh,
        loc=scenes.loc,
        split=split_n,
        bear_id=token,
        bear_scale=bear_scale,
        bear_translate=bear_translate,
        record_index=scenes.record_index,
        split_sample_ids=split_sample_ids,
        marker_kw=marker_kw,
        multiprocessing=multiprocessing,
    )
    layer = {"bear_id": token, "split": split_n, "template": template}
    scenes.marker_layers = [
        existing
        for existing in scenes.marker_layers
        if not (
            existing["bear_id"] == token and existing["split"] == split_n
        )
    ]
    scenes.marker_layers.append(layer)
    _rewrite_epoch_povs(scenes)
    return scenes


def add_loss_board_to_figure3_scenes(
    scenes: Figure3SceneSet,
    combined: Mapping[str, Any],
    *,
    board_id: str = "loss",
    width_mm: float | None = None,
    aspect: float = 0.69,
    thickness_mm: float = 0.7,
    center: Sequence[float] | None = None,
    center_fwd_frac: float = 0.44,
    center_right_frac: float = -0.33,
    center_up_frac: float = 0.22,
    label_font_px: int | None = None,
    curve_width_px: int | None = None,
    dot_radius_mm: float | None = None,
) -> Figure3SceneSet:
    """Add a static chalkboard object plus per-epoch luminous loss dots.

    ``center_right_frac`` is signed along the camera-right axis: negative shifts
    the board to the viewer's left, positive to the right (fraction of camera
    standoff distance).

    ``label_font_px`` sets the FreeType pixel size for the axis end-tick labels
    on the board PNG (``None`` → automatic ~0.12× texture height).

    ``curve_width_px`` sets the Pillow stroke width for the red/blue loss curves
    (``None`` → automatic ~0.054× texture width). ``dot_radius_mm`` is the
    luminous marker radius on the board face (``None`` → ~2.5% of the shorter
    board edge).
    """
    dist = float(np.linalg.norm(np.asarray(scenes.look) - np.asarray(scenes.loc)))
    board_w = max(12.0, 0.26 * dist) if width_mm is None else float(width_mm)
    board_h = max(8.0, float(board_w) * float(aspect))
    fwd, right, up = _camera_basis(scenes.loc, scenes.look)
    board_center = (
        np.asarray(center, dtype=float).reshape(3)
        if center is not None
        else (
            np.asarray(scenes.loc, dtype=float)
            + float(center_fwd_frac) * dist * fwd
            + float(center_right_frac) * dist * right
            + float(center_up_frac) * dist * up
        )
    )
    tex_dir = scenes.scenes_dir / FIGURE3_BOARD_TEXTURE_DIR
    tex_dir.mkdir(parents=True, exist_ok=True)
    _write_figure3_loss_board_png(
        tex_dir / FIGURE3_BOARD_TEXTURE_NAME,
        combined=combined,
        label_font_px=label_font_px,
        curve_width_px=curve_width_px,
    )
    object_name = (
        FIGURE3_BOARD_OBJECT_INC
        if _layer_token(board_id) == "loss"
        else f"figure3_board_{_layer_token(board_id)}_object.inc"
    )
    object_text = _figure3_loss_board_pov_text(
        png_name=f"{FIGURE3_BOARD_TEXTURE_DIR}/{FIGURE3_BOARD_TEXTURE_NAME}",
        loc=scenes.loc,
        look=scenes.look,
        board_center=board_center,
        half_width=0.5 * board_w,
        half_height=0.5 * board_h,
        thickness=thickness_mm,
    )
    (scenes.scenes_dir / object_name).write_text(object_text, encoding="utf-8")
    _append_include(scenes, object_name)
    template = _board_inc_template(board_id=board_id)
    num_epochs = int(combined.get("num_epochs", max(scenes.epochs) if scenes.epochs else 0))
    for epoch in scenes.epochs:
        text = _figure3_loss_board_dot_pov_text(
            epoch=int(epoch),
            num_epochs=num_epochs,
            combined=combined,
            loc=scenes.loc,
            look=scenes.look,
            board_center=board_center,
            half_width=0.5 * board_w,
            half_height=0.5 * board_h,
            thickness=thickness_mm,
            label_font_px=label_font_px,
            dot_radius_mm=dot_radius_mm,
        )
        name = template.format(epoch=int(epoch))
        (scenes.scenes_dir / name).write_text(text, encoding="utf-8")
    layer = {"board_id": _layer_token(board_id), "template": template}
    scenes.board_layers = [
        existing
        for existing in scenes.board_layers
        if existing["board_id"] != layer["board_id"]
    ]
    scenes.board_layers.append(layer)
    _rewrite_epoch_povs(scenes)
    return scenes


def write_figure3_pov_scenes(
    combined: Mapping[str, Any],
    *,
    layout: Mapping[str, Path],
    repo: Path,
    epochs: Sequence[int] | None = None,
    stl_path: Path | None = None,
    sample_id: str | None = None,
    camera_location: Sequence[float] | None = None,
    camera_look_at: Sequence[float] | None = None,
    fov_deg: float | None = None,
    distance_scale: float | None = None,
    yaw_deg: float | None = None,
    light_intensity: float | None = None,
    ball_lights: bool | None = None,
    ball_spotlights: bool | None = None,
    ball_spotlight_angle_deg: float | None = None,
    ball_light_intensity: float | None = None,
    draw_prediction_links: bool | None = None,
    distance_luminosity: bool | None = None,
    distance_transparency: bool | None = None,
    green_luminosity: float | None = None,
    pred_luminosity_at_target: float | None = None,
    pooling_luminosity_at_target: float | None = None,
    fourier_luminosity_at_target: float | None = None,
    pred_luminosity_hold_mm: float | None = None,
    pred_luminosity_zero_mm: float | None = None,
    pred_transmit_hold_mm: float | None = None,
    pred_transmit_far: float | None = None,
    pred_transmit_zero_mm: float | None = None,
    pred_transmit_at_target: float | None = None,
) -> tuple[list[Path], dict[str, list[float]]]:
    """Write world + bear + markers. Convenience wrapper around the staged API."""
    num_epochs = int(combined.get("num_epochs", 0))
    if epochs is None:
        epochs = list(range(0, num_epochs + 1))
    marker_kw = {
        k: v
        for k, v in {
            "ball_lights": ball_lights,
            "ball_spotlights": ball_spotlights,
            "ball_spotlight_angle_deg": ball_spotlight_angle_deg,
            "ball_light_intensity": ball_light_intensity,
            "draw_prediction_links": draw_prediction_links,
            "distance_luminosity": distance_luminosity,
            "distance_transparency": distance_transparency,
            "green_luminosity": green_luminosity,
            "pred_luminosity_at_target": pred_luminosity_at_target,
            "pooling_luminosity_at_target": pooling_luminosity_at_target,
            "fourier_luminosity_at_target": fourier_luminosity_at_target,
            "pred_luminosity_hold_mm": pred_luminosity_hold_mm,
            "pred_luminosity_zero_mm": pred_luminosity_zero_mm,
            "pred_transmit_hold_mm": pred_transmit_hold_mm,
            "pred_transmit_far": pred_transmit_far,
            "pred_transmit_zero_mm": pred_transmit_zero_mm,
            "pred_transmit_at_target": pred_transmit_at_target,
        }.items()
        if v is not None
    }
    scenes = generate_figure3_scenes(
        layout=layout,
        repo=repo,
        epochs=epochs,
        combined=combined,
        stl_path=stl_path,
        sample_id=sample_id,
        camera_location=camera_location,
        camera_look_at=camera_look_at,
        fov_deg=fov_deg,
        distance_scale=distance_scale,
        yaw_deg=yaw_deg,
        light_intensity=light_intensity,
    )
    add_bear_to_figure3_scenes(scenes)
    add_markers_to_figure3_scenes(scenes, combined, **marker_kw)
    add_loss_board_to_figure3_scenes(scenes, combined)
    return scenes.pov_paths, scenes.camera


def render_figure3_frames(
    pov_paths: Sequence[Path],
    *,
    renders_dir: Path,
    width: int = 960,
    height: int = 720,
    multiprocessing: bool = True,
) -> tuple[list[Path], list[str]]:
    renders_dir = Path(renders_dir)
    renders_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(Path(pov), renders_dir / f"{Path(pov).stem}.png") for pov in pov_paths]
    notes: list[str] = []
    if not jobs:
        return [], notes

    workers = figure3_render_worker_count(len(jobs), multiprocessing=multiprocessing)
    # One POV-Ray thread per process so n-2 workers use about n-2 cores.
    work_threads = 1 if workers > 1 else None
    if workers > 1:
        print(
            f"Rendering {len(jobs)} POV frames with {workers} processes "
            "(+WT1 each).",
            flush=True,
        )

    rendered: list[Path | None]
    if workers == 1:
        rendered = [
            render_pov_file(pov, png, width=width, height=height, work_threads=work_threads)
            for pov, png in jobs
        ]
    else:
        mp_ctx = _figure3_process_pool_context()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
            rendered = [
                Path(out) if out else None
                for out in pool.map(
                    _render_one_pov_png,
                    [str(pov) for pov, _png in jobs],
                    [str(png) for _pov, png in jobs],
                    [width] * len(jobs),
                    [height] * len(jobs),
                    [work_threads] * len(jobs),
                )
            ]

    pngs: list[Path] = []
    for (pov, png), out in zip(jobs, rendered, strict=True):
        if out is None:
            notes.append(
                "POV-Ray not found on PATH. Scene files were written; render with:\n"
                f"  cd {display_path(renders_dir)} && "
                f"povray +I{pov.resolve()} +O{png.name} +L{pov.parent} "
                f"+W{width} +H{height} +FN -D"
            )
            break
        pngs.append(out)
    return pngs, notes


def assemble_gif(png_paths: Sequence[Path], gif_path: Path, *, duration_ms: int = 80) -> Path:
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.open(p).convert("RGB") for p in png_paths]
    if not frames:
        raise ValueError("no PNG frames to assemble into a GIF")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
    )
    return gif_path


# Default post-render captions under each bear (keyed by ``bear_id`` token).
FIGURE3_DEFAULT_BEAR_LABELS: dict[str, str] = {
    "train": "Training",
    "training": "Training",
    "validation": "Validation",
    "val": "Validation",
    "default": "Test",
    "test": "Test",
}


def _project_world_to_pixel(
    point: Sequence[float] | np.ndarray,
    *,
    loc: np.ndarray,
    look: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """POV-Ray-like perspective: ``angle`` = horizontal FOV, ``right x*(w/h)``."""
    p = np.asarray(point, dtype=float).reshape(3)
    fwd, right, up = _camera_basis(loc, look)
    d = p - np.asarray(loc, dtype=float).reshape(3)
    depth = float(np.dot(d, fwd))
    if depth <= 1e-6:
        return None
    x = float(np.dot(d, right))
    y = float(np.dot(d, up))
    t = float(np.tan(np.deg2rad(float(fov_deg)) * 0.5))
    if t <= 1e-9:
        return None
    aspect = float(width) / max(float(height), 1e-9)
    px = 0.5 * float(width) * (1.0 + (x / depth) / t)
    py = 0.5 * float(height) * (1.0 - (y / depth) * aspect / t)
    return px, py


def _figure3_bear_label_world_point(
    *,
    mesh: Any,
    scale: float,
    translate: Sequence[float] | None,
) -> np.ndarray:
    """Point just under the placed bear AABB (horizontal centre, below feet)."""
    lo, hi, span = figure3_mesh_metrics(mesh)
    corners = np.array(
        [
            [x, y, z]
            for x in (float(lo[0]), float(hi[0]))
            for y in (float(lo[1]), float(hi[1]))
            for z in (float(lo[2]), float(hi[2]))
        ],
        dtype=float,
    )
    placed = np.asarray(
        [
            figure3_apply_bear_placement(
                c, mesh=mesh, scale=float(scale), translate=translate
            )
            for c in corners
        ],
        dtype=float,
    )
    zmin = float(placed[:, 2].min())
    sc = max(float(scale), 1e-9)
    return np.array(
        [
            float(placed[:, 0].mean()),
            float(placed[:, 1].mean()),
            zmin - 0.08 * float(span) * sc,
        ],
        dtype=float,
    )


def figure3_bear_label_pixel_anchors(
    scenes: Figure3SceneSet,
    *,
    width: int,
    height: int,
    labels: Mapping[str, str] | None = None,
) -> list[tuple[str, float, float]]:
    """``(caption, px, py)`` for each labelled bear, left-to-right on screen."""
    label_map = {
        **FIGURE3_DEFAULT_BEAR_LABELS,
        **({str(k): str(v) for k, v in labels.items()} if labels else {}),
    }
    anchors: list[tuple[str, float, float]] = []
    for token, info in (scenes.bears or {}).items():
        caption = label_map.get(str(token))
        if not caption:
            continue
        mesh = info.get("mesh")
        if mesh is None:
            continue
        world = _figure3_bear_label_world_point(
            mesh=mesh,
            scale=float(info.get("scale", 1.0)),
            translate=info.get("translate"),
        )
        pix = _project_world_to_pixel(
            world,
            loc=scenes.loc,
            look=scenes.look,
            fov_deg=float(scenes.fov_deg),
            width=int(width),
            height=int(height),
        )
        if pix is None:
            continue
        anchors.append((caption, float(pix[0]), float(pix[1])))
    anchors.sort(key=lambda item: item[1])
    return anchors


def annotate_figure3_bear_labels(
    png_paths: Sequence[Path],
    scenes: Figure3SceneSet,
    *,
    labels: Mapping[str, str] | None = None,
    font_px: int | None = None,
) -> list[Path]:
    """Burn Training / Validation / Test captions under bears on each PNG."""
    paths = [Path(p) for p in png_paths]
    if not paths or not scenes.bears:
        return paths
    sample = Image.open(paths[0])
    width, height = sample.size
    sample.close()
    anchors = figure3_bear_label_pixel_anchors(
        scenes, width=width, height=height, labels=labels
    )
    if not anchors:
        return paths
    size = max(18, int(font_px) if font_px is not None else int(round(0.035 * height)))
    font = _sans(size)
    ink = (245, 245, 240)
    shadow = (8, 10, 14)
    for path in paths:
        image = Image.open(path).convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        for caption, px, py in anchors:
            # Slightly below the projected foot; keep on-canvas.
            x = float(np.clip(px, 8.0, width - 8.0))
            y = float(np.clip(py + 0.012 * height, 8.0, height - 8.0))
            for dx, dy in ((2, 2), (1, 1)):
                draw.text(
                    (x + dx, y + dy),
                    caption,
                    fill=(*shadow, 200),
                    font=font,
                    anchor="mt",
                )
            draw.text((x, y), caption, fill=(*ink, 255), font=font, anchor="mt")
        image.convert("RGB").save(path)
    return paths


def prepare_figure3_history(
    *,
    repo_root_path: Path | None = None,
    output_root: Path | None = None,
    workbook_path: Path | None = None,
    data_root: Path | None = None,
    num_epochs: int = 200,
    batch_size: int = 32,
    seed: int = 0,
    n_tracked: int | None = None,
    skip_train_if_history_exists: bool = True,
    force_retrain: bool = False,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Train or reuse JSON histories. Returns ``(layout, combined)``."""
    root = repo_root() if repo_root_path is None else Path(repo_root_path)
    fig_root = (
        default_figure3_root(root) if output_root is None else Path(output_root)
    )
    layout = figure3_layout(fig_root)
    for key in ("root", "scenes", "renders", "final"):
        Path(layout[key]).mkdir(parents=True, exist_ok=True)

    combined_path = layout["combined"]
    have_history = combined_path.is_file() and layout["pooling"].is_file()
    if force_retrain or not (skip_train_if_history_exists and have_history):
        train_figure3_convergence(
            repo=root,
            workbook_path=workbook_path,
            data_root=data_root,
            output_root=fig_root,
            num_epochs=num_epochs,
            batch_size=batch_size,
            seed=seed,
            n_tracked=n_tracked,
            device=device,
            verbose=verbose,
        )
    elif verbose:
        print(f"Reusing history {display_path(combined_path)}", flush=True)
    return layout, load_history(combined_path)


def render_figure3_scenes(
    scenes: Figure3SceneSet,
    *,
    render: bool = True,
    multiprocessing: bool = True,
    gif_duration_ms: int = 80,
    width: int = 960,
    height: int = 720,
    bear_labels: Mapping[str, str] | None = None,
    bear_label_font_px: int | None = None,
) -> tuple[list[Path], list[str]]:
    """Ray-trace the prepared ``.pov`` files, GIF, and final still.

    After POV-Ray finishes, burn bear captions (Training / Validation / Test
    by default) into each PNG under the corresponding mesh.
    """
    notes: list[str] = []
    pngs: list[Path] = []
    if not render:
        return pngs, notes
    pngs, notes = render_figure3_frames(
        scenes.pov_paths,
        renders_dir=Path(scenes.layout["renders"]),
        width=width,
        height=height,
        multiprocessing=multiprocessing,
    )
    if pngs:
        annotate_figure3_bear_labels(
            pngs,
            scenes,
            labels=bear_labels,
            font_px=bear_label_font_px,
        )
        assemble_gif(
            pngs, Path(scenes.layout["final_gif"]), duration_ms=gif_duration_ms
        )
        shutil.copy2(pngs[-1], Path(scenes.layout["final_png"]))
    return pngs, notes


def export_figure3_convergence(
    *,
    repo_root_path: Path | None = None,
    output_root: Path | None = None,
    workbook_path: Path | None = None,
    data_root: Path | None = None,
    num_epochs: int = 200,
    batch_size: int = 32,
    seed: int = 0,
    n_tracked: int | None = None,
    skip_train_if_history_exists: bool = True,
    force_retrain: bool = False,
    render: bool = True,
    render_epoch_stride: int = 1,
    gif_duration_ms: int = 80,
    device: str | None = None,
    stl_path: Path | None = None,
    camera_location: Sequence[float] | None = None,
    camera_look_at: Sequence[float] | None = None,
    fov_deg: float | None = None,
    camera_distance_scale: float | None = None,
    camera_yaw_deg: float | None = None,
    light_intensity: float | None = None,
    ball_lights: bool | None = None,
    ball_spotlights: bool | None = None,
    ball_spotlight_angle_deg: float | None = None,
    ball_light_intensity: float | None = None,
    draw_prediction_links: bool | None = None,
    distance_luminosity: bool | None = None,
    distance_transparency: bool | None = None,
    green_luminosity: float | None = None,
    pred_luminosity_at_target: float | None = None,
    pooling_luminosity_at_target: float | None = None,
    fourier_luminosity_at_target: float | None = None,
    pred_luminosity_hold_mm: float | None = None,
    pred_luminosity_zero_mm: float | None = None,
    pred_transmit_hold_mm: float | None = None,
    pred_transmit_far: float | None = None,
    pred_transmit_zero_mm: float | None = None,
    pred_transmit_at_target: float | None = None,
    multiprocessing: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full Figure 3 pipeline: history → world → bear → markers → render.

    Notebooks should call ``prepare_figure3_history``, ``generate_figure3_scenes``,
    ``add_bear_to_figure3_scenes``, ``add_markers_to_figure3_scenes``, and
    ``render_figure3_scenes`` sequentially when iterating on the illustration.
    """
    layout, combined = prepare_figure3_history(
        repo_root_path=repo_root_path,
        output_root=output_root,
        workbook_path=workbook_path,
        data_root=data_root,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        n_tracked=n_tracked,
        skip_train_if_history_exists=skip_train_if_history_exists,
        force_retrain=force_retrain,
        device=device,
        verbose=verbose,
    )
    root = repo_root() if repo_root_path is None else Path(repo_root_path)
    sample_id, p_err, f_err, adv = select_best_fourier_advantage_sample(combined)
    num_ep = int(combined.get("num_epochs", num_epochs))
    epochs = figure3_epoch_list(
        combined, num_epochs=num_ep, stride=render_epoch_stride
    )
    marker_kw = {
        k: v
        for k, v in {
            "ball_lights": ball_lights,
            "ball_spotlights": ball_spotlights,
            "ball_spotlight_angle_deg": ball_spotlight_angle_deg,
            "ball_light_intensity": ball_light_intensity,
            "draw_prediction_links": draw_prediction_links,
            "distance_luminosity": distance_luminosity,
            "distance_transparency": distance_transparency,
            "green_luminosity": green_luminosity,
            "pred_luminosity_at_target": pred_luminosity_at_target,
            "pooling_luminosity_at_target": pooling_luminosity_at_target,
            "fourier_luminosity_at_target": fourier_luminosity_at_target,
            "pred_luminosity_hold_mm": pred_luminosity_hold_mm,
            "pred_luminosity_zero_mm": pred_luminosity_zero_mm,
            "pred_transmit_hold_mm": pred_transmit_hold_mm,
            "pred_transmit_far": pred_transmit_far,
            "pred_transmit_zero_mm": pred_transmit_zero_mm,
            "pred_transmit_at_target": pred_transmit_at_target,
        }.items()
        if v is not None
    }
    scenes = generate_figure3_scenes(
        layout=layout,
        repo=root,
        epochs=epochs,
        combined=combined,
        stl_path=stl_path,
        camera_location=camera_location,
        camera_look_at=camera_look_at,
        fov_deg=fov_deg,
        distance_scale=camera_distance_scale,
        yaw_deg=camera_yaw_deg,
        light_intensity=light_intensity,
    )
    add_bear_to_figure3_scenes(scenes)
    add_markers_to_figure3_scenes(scenes, combined, **marker_kw)
    add_loss_board_to_figure3_scenes(scenes, combined)
    pov_paths, camera = scenes.pov_paths, scenes.camera
    pngs, notes = render_figure3_scenes(
        scenes,
        render=render,
        multiprocessing=multiprocessing,
        gif_duration_ms=gif_duration_ms,
    )

    summary = {
        "n_tracked_validation": len(tracked_sample_ids(combined)),
        "sample_id_best_fourier_advantage": sample_id,
        "pooled_final_error": p_err,
        "fourier_final_error": f_err,
        "fourier_advantage": adv,
        "prediction_history": {
            "pooling": str(display_path(layout["pooling"])),
            "fourier": str(display_path(layout["fourier"])),
            "combined": str(display_path(layout["combined"])),
        },
        "pov_scenes_dir": str(display_path(layout["scenes"])),
        "renders_dir": str(display_path(layout["renders"])),
        "final_png": str(display_path(layout["final_png"])),
        "final_gif": str(display_path(layout["final_gif"])),
        "n_pov_scenes": len(pov_paths),
        "n_renders": len(pngs),
        "render_multiprocessing": bool(multiprocessing),
        "render_workers": figure3_render_worker_count(
            len(pov_paths), multiprocessing=multiprocessing
        ),
        "camera_location": camera["location"],
        "camera_look_at": camera["look_at"],
        "light_intensity": (
            1.0 if light_intensity is None else float(light_intensity)
        ),
        "ball_lights": False if ball_lights is None else bool(ball_lights),
        "ball_spotlights": (
            DEFAULT_BALL_SPOTLIGHTS if ball_spotlights is None else bool(ball_spotlights)
        ),
        "ball_spotlight_angle_deg": (
            DEFAULT_BALL_SPOTLIGHT_ANGLE_DEG
            if ball_spotlight_angle_deg is None
            else float(ball_spotlight_angle_deg)
        ),
        "ball_light_intensity": (
            DEFAULT_BALL_LIGHT_INTENSITY
            if ball_light_intensity is None
            else float(ball_light_intensity)
        ),
        "draw_prediction_links": (
            False if draw_prediction_links is None else bool(draw_prediction_links)
        ),
        "distance_luminosity": (
            False if distance_luminosity is None else bool(distance_luminosity)
        ),
        "distance_transparency": (
            True if distance_transparency is None else bool(distance_transparency)
        ),
        "green_luminosity": (
            1.0 / 3.0 if green_luminosity is None else float(green_luminosity)
        ),
        "pred_luminosity_at_target": (
            1.0
            if pred_luminosity_at_target is None
            else float(pred_luminosity_at_target)
        ),
        "pooling_luminosity_at_target": (
            (
                1.0 if pred_luminosity_at_target is None else float(pred_luminosity_at_target)
            )
            if pooling_luminosity_at_target is None
            else float(pooling_luminosity_at_target)
        ),
        "fourier_luminosity_at_target": (
            (
                1.0 if pred_luminosity_at_target is None else float(pred_luminosity_at_target)
            )
            if fourier_luminosity_at_target is None
            else float(fourier_luminosity_at_target)
        ),
        "pred_luminosity_hold_mm": (
            1.0 if pred_luminosity_hold_mm is None else float(pred_luminosity_hold_mm)
        ),
        "pred_luminosity_zero_mm": (
            10.0
            if pred_luminosity_zero_mm is None
            else float(pred_luminosity_zero_mm)
        ),
        "pred_transmit_hold_mm": (
            1.0 if pred_transmit_hold_mm is None else float(pred_transmit_hold_mm)
        ),
        "pred_transmit_far": (
            0.95 if pred_transmit_far is None else float(pred_transmit_far)
        ),
        "pred_transmit_zero_mm": (
            10.0 if pred_transmit_zero_mm is None else float(pred_transmit_zero_mm)
        ),
        "pred_transmit_at_target": (
            0.0
            if pred_transmit_at_target is None
            else float(pred_transmit_at_target)
        ),
        "notes": notes,
    }
    layout["summary_json"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if verbose:
        print("=== Figure 3 summary ===", flush=True)
        for k, v in summary.items():
            if k == "notes":
                continue
            print(f"  {k}: {v}", flush=True)
        for note in notes:
            print(note, flush=True)
        print(
            f"Best demo sample (Fourier advantage): {sample_id}  "
            f"pooled_err={p_err:.3f}  fourier_err={f_err:.3f}  "
            f"advantage={adv:.3f}",
            flush=True,
        )
    return {"layout": layout, "summary": summary, "combined": combined}
