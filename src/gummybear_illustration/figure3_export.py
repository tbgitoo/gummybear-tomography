"""Export Figure 3 POV frames, renders, GIF, and final still from JSON history."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from gummybear.paths import display_path

from .export_m8_physical_scene import render_pov_file
from .figure3_history import (
    MODEL_FOURIER,
    MODEL_POOLING,
    default_figure3_root,
    history_paths,
    load_history,
    records_for_sample,
    select_best_fourier_advantage_sample,
)
from .figure3_pov_scene import (
    build_figure3_epoch_pov,
    figure3_camera_pose,
    prepare_bear_inc,
    tracked_sample_ids,
)
from .figure3_train import train_figure3_convergence
from .network_captions import COLOR_FOURIER, COLOR_GAP, COLOR_TARGET
from .paths import default_cad_dir, repo_root


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


def _overlay_figure3_captions_for_frame(
    png_path: Path,
    pov_path: Path,
    caption_meta: Mapping[str, Any],
) -> None:
    epoch = int(pov_path.stem.rsplit("_", 1)[-1])
    combined = caption_meta["combined"]
    sids = tracked_sample_ids(combined)
    p_errs: list[float] = []
    f_errs: list[float] = []
    for sid in sids:
        p_recs = records_for_sample(combined, sid, model_type=MODEL_POOLING)
        f_recs = records_for_sample(combined, sid, model_type=MODEL_FOURIER)
        p_errs.append(
            next(
                (
                    float(r["localization_error"])
                    for r in reversed(p_recs)
                    if int(r["epoch"]) <= epoch
                ),
                float("nan"),
            )
        )
        f_errs.append(
            next(
                (
                    float(r["localization_error"])
                    for r in reversed(f_recs)
                    if int(r["epoch"]) <= epoch
                ),
                float("nan"),
            )
        )
    overlay_figure3_captions(
        png_path,
        epoch=epoch,
        num_epochs=int(caption_meta.get("num_epochs", 0)),
        n_val=len(sids),
        pooled_err=float(np.nanmean(p_errs)) if p_errs else float("nan"),
        fourier_err=float(np.nanmean(f_errs)) if f_errs else float("nan"),
    )


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


def overlay_figure3_captions(
    png_path: Path,
    *,
    epoch: int,
    num_epochs: int,
    n_val: int,
    pooled_err: float,
    fourier_err: float,
) -> Path:
    image = Image.open(png_path).convert("RGB")
    w, h = image.size
    draw = ImageDraw.Draw(image)
    font = _sans(max(14, int(0.028 * h)))
    small = _sans(max(12, int(0.022 * h)))
    lines = [
        ((0.50, 0.06), "Figure 3 — pooling vs Fourier localization", (0, 0, 0), font),
        ((0.18, 0.94), "average pooling", COLOR_GAP, small),
        ((0.50, 0.94), "target (validation)", COLOR_TARGET, small),
        ((0.82, 0.94), "Fourier pooling", COLOR_FOURIER, small),
        (
            (0.50, 0.98),
            f"epoch {epoch}/{num_epochs}   n_val={n_val}   "
            f"mean_err_pool={pooled_err:.2f}  mean_err_fourier={fourier_err:.2f}",
            (40, 40, 40),
            small,
        ),
    ]
    for (xf, yf), text, fill, fnt in lines:
        draw.text((xf * w, yf * h), text, font=fnt, fill=fill, anchor="ms")
    image.save(png_path)
    return png_path


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
    scenes_dir = Path(layout["scenes"])
    scenes_dir.mkdir(parents=True, exist_ok=True)
    sid = sample_id
    if sid is None:
        tracked = combined.get("tracked_samples") or []
        sid = str(tracked[0]["sample_id"]) if tracked else ""
    stl = _resolve_stl(
        repo=repo, combined=combined, sample_id=str(sid), stl_path=stl_path
    )
    inc_path = scenes_dir / "figure3_bear.inc"
    bear_inc, mesh = prepare_bear_inc(stl, inc_path)
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
    camera = {
        "location": [float(v) for v in loc],
        "look_at": [float(v) for v in look],
    }
    build_kw: dict[str, Any] = {
        "camera_location": loc,
        "camera_look_at": look,
    }
    if fov_deg is not None:
        build_kw["fov_deg"] = float(fov_deg)
    if distance_scale is not None:
        build_kw["distance_scale"] = float(distance_scale)
    if yaw_deg is not None:
        build_kw["yaw_deg"] = float(yaw_deg)
    if light_intensity is not None:
        build_kw["light_intensity"] = float(light_intensity)
    if ball_lights is not None:
        build_kw["ball_lights"] = bool(ball_lights)
    if draw_prediction_links is not None:
        build_kw["draw_prediction_links"] = bool(draw_prediction_links)
    if distance_luminosity is not None:
        build_kw["distance_luminosity"] = bool(distance_luminosity)
    if distance_transparency is not None:
        build_kw["distance_transparency"] = bool(distance_transparency)
    if green_luminosity is not None:
        build_kw["green_luminosity"] = float(green_luminosity)
    if pred_luminosity_at_target is not None:
        build_kw["pred_luminosity_at_target"] = float(pred_luminosity_at_target)
    if pooling_luminosity_at_target is not None:
        build_kw["pooling_luminosity_at_target"] = float(pooling_luminosity_at_target)
    if fourier_luminosity_at_target is not None:
        build_kw["fourier_luminosity_at_target"] = float(fourier_luminosity_at_target)
    if pred_luminosity_hold_mm is not None:
        build_kw["pred_luminosity_hold_mm"] = float(pred_luminosity_hold_mm)
    if pred_luminosity_zero_mm is not None:
        build_kw["pred_luminosity_zero_mm"] = float(pred_luminosity_zero_mm)
    if pred_transmit_hold_mm is not None:
        build_kw["pred_transmit_hold_mm"] = float(pred_transmit_hold_mm)
    if pred_transmit_far is not None:
        build_kw["pred_transmit_far"] = float(pred_transmit_far)
    if pred_transmit_zero_mm is not None:
        build_kw["pred_transmit_zero_mm"] = float(pred_transmit_zero_mm)
    if pred_transmit_at_target is not None:
        build_kw["pred_transmit_at_target"] = float(pred_transmit_at_target)
    num_epochs = int(combined.get("num_epochs", 0))
    if epochs is None:
        epochs = list(range(0, num_epochs + 1))
    written: list[Path] = []
    for epoch in epochs:
        pov = scenes_dir / f"figure3_epoch_{int(epoch):04d}.pov"
        text = build_figure3_epoch_pov(
            combined_history=combined,
            epoch=int(epoch),
            bear_inc_name=bear_inc,
            mesh=mesh,
            **build_kw,
        )
        pov.write_text(text, encoding="utf-8")
        written.append(pov)
    return written, camera


def render_figure3_frames(
    pov_paths: Sequence[Path],
    *,
    renders_dir: Path,
    width: int = 960,
    height: int = 720,
    caption_meta: Mapping[str, Any] | None = None,
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
        with ProcessPoolExecutor(max_workers=workers) as pool:
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
        if caption_meta:
            _overlay_figure3_captions_for_frame(out, pov, caption_meta)
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
    """Full Figure 3 pipeline: train→JSON (optional) → POV → render → GIF."""
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

    combined = load_history(combined_path)
    sample_id, p_err, f_err, adv = select_best_fourier_advantage_sample(combined)
    num_ep = int(combined.get("num_epochs", num_epochs))
    stride = max(int(render_epoch_stride), 1)
    epochs = list(range(0, num_ep + 1, stride))
    if epochs[-1] != num_ep:
        epochs.append(num_ep)

    pov_paths, camera = write_figure3_pov_scenes(
        combined,
        layout=layout,
        repo=root,
        epochs=epochs,
        stl_path=stl_path,
        camera_location=camera_location,
        camera_look_at=camera_look_at,
        fov_deg=fov_deg,
        distance_scale=camera_distance_scale,
        yaw_deg=camera_yaw_deg,
        light_intensity=light_intensity,
        ball_lights=ball_lights,
        draw_prediction_links=draw_prediction_links,
        distance_luminosity=distance_luminosity,
        distance_transparency=distance_transparency,
        green_luminosity=green_luminosity,
        pred_luminosity_at_target=pred_luminosity_at_target,
        pooling_luminosity_at_target=pooling_luminosity_at_target,
        fourier_luminosity_at_target=fourier_luminosity_at_target,
        pred_luminosity_hold_mm=pred_luminosity_hold_mm,
        pred_luminosity_zero_mm=pred_luminosity_zero_mm,
        pred_transmit_hold_mm=pred_transmit_hold_mm,
        pred_transmit_far=pred_transmit_far,
        pred_transmit_zero_mm=pred_transmit_zero_mm,
        pred_transmit_at_target=pred_transmit_at_target,
    )
    notes: list[str] = []
    pngs: list[Path] = []
    if render:
        pngs, notes = render_figure3_frames(
            pov_paths,
            renders_dir=layout["renders"],
            caption_meta={
                "combined": combined,
                "num_epochs": num_ep,
            },
            multiprocessing=multiprocessing,
        )
        if pngs:
            assemble_gif(pngs, layout["final_gif"], duration_ms=gif_duration_ms)
            shutil.copy2(pngs[-1], layout["final_png"])

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
