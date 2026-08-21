"""POV-Ray scenes for Figure 3: prediction convergence inside the bear."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import trimesh

from .figure3_history import (
    MODEL_FOURIER,
    MODEL_POOLING,
    Figure3RecordIndex,
    records_for_sample,
)
from .pov_primitives import comment_block, cylinder, plane_z, pov_vec, sphere
from .pov_scene import bear_triangle_edge_cylinders, sky_and_horizon
from .stl_to_mesh2 import write_stl_mesh2_inc

# Match network-scene branch colours (dark red / dark blue) + green target.
POOLING_RGB = "<0.58, 0.10, 0.12>"
FOURIER_RGB = "<0.14, 0.34, 0.78>"
TARGET_RGB = "<0.14, 0.82, 0.28>"
POOLING_LIGHT = (2.35, 0.32, 0.22)
FOURIER_LIGHT = (0.28, 0.55, 2.40)
TARGET_LIGHT = (0.28, 1.90, 0.42)
# Shared critical distance (mm): luminosity → 0 / transparency → far value.
DEFAULT_PRED_APPROACH_ZERO_MM = 10.0
DEFAULT_PRED_HOLD_MM = 1.0
DEFAULT_PRED_LUMINOSITY_AT_TARGET = 1.0
DEFAULT_GREEN_LUMINOSITY = 1.0 / 3.0
# POV transmit: 0 = opaque, 1 = invisible. Far preds default mostly transparent.
DEFAULT_PRED_TRANSMIT_FAR = 0.95
DEFAULT_PRED_TRANSMIT_AT_TARGET = 0.0
# When a prediction sits inside the green ball, open the target so the light escapes.
DEFAULT_TARGET_HIT_TRANSMIT = 0.9
DEFAULT_BALL_SPOTLIGHTS = True
DEFAULT_BALL_SPOTLIGHT_ANGLE_DEG = 15.0
# Scales predicted-ball ``light_source`` colour only (not marker emission,
# not the scene key/fill). Spotlights need this well above 1 so the cone
# reads as a local patch on the bear rather than a faint wash.
DEFAULT_BALL_LIGHT_INTENSITY = 8.0
# Back-compat alias
DEFAULT_PRED_LUMINOSITY_ZERO_MM = DEFAULT_PRED_APPROACH_ZERO_MM
# Marker emission/light baseline. Kept modest so markers do not overpower the
# bear; the notebook ``LIGHT_INTENSITY`` controls scene lighting separately.
LIGHT_I_MAX = 0.18
DEFAULT_LIGHT_INTENSITY = 1.0


def simulation_xyz_to_pov(xyz: Sequence[float]) -> np.ndarray:
    """Identity map: catalog xyz are already simulation millimetres (z-up)."""
    return np.asarray(xyz, dtype=float).reshape(3)


def _record_at_epoch(recs: Sequence[Mapping[str, Any]], epoch: int) -> dict[str, Any]:
    exact = [r for r in recs if int(r["epoch"]) == int(epoch)]
    if exact:
        return dict(exact[0])
    below = [r for r in recs if int(r["epoch"]) <= int(epoch)]
    if not below:
        return dict(recs[0])
    return dict(max(below, key=lambda r: int(r["epoch"])))


def distance_luminosity(
    dist_mm: float,
    *,
    hold_mm: float = DEFAULT_PRED_HOLD_MM,
    zero_mm: float = DEFAULT_PRED_APPROACH_ZERO_MM,
    at_target: float = 1.0,
) -> float:
    """Plateau then linear ramp: hold until ``hold_mm``, 0 at/beyond ``zero_mm``."""
    d = max(float(dist_mm), 0.0)
    h = max(float(hold_mm), 0.0)
    z = max(float(zero_mm), h + 1e-9)
    if d <= h:
        return max(0.0, float(at_target))
    t = 1.0 - (d - h) / (z - h)
    if t <= 0.0:
        return 0.0
    return max(0.0, float(at_target) * t)


def approach_weight(
    err_mm: float,
    *,
    hold_mm: float = DEFAULT_PRED_HOLD_MM,
    zero_mm: float = DEFAULT_PRED_APPROACH_ZERO_MM,
) -> float:
    """0–1 approach factor: 1 on target, 0 at/beyond ``zero_mm``."""
    return distance_luminosity(
        err_mm, hold_mm=hold_mm, zero_mm=zero_mm, at_target=1.0
    )


def distance_transmit(
    dist_mm: float,
    *,
    hold_mm: float = DEFAULT_PRED_HOLD_MM,
    zero_mm: float = DEFAULT_PRED_APPROACH_ZERO_MM,
    transmit_far: float = DEFAULT_PRED_TRANSMIT_FAR,
    transmit_at_target: float = DEFAULT_PRED_TRANSMIT_AT_TARGET,
) -> float:
    """POV transmit: hold near value until ``hold_mm``, then decay to ``transmit_far``."""
    w = approach_weight(dist_mm, hold_mm=hold_mm, zero_mm=zero_mm)
    far = float(np.clip(transmit_far, 0.0, 1.0))
    near = float(np.clip(transmit_at_target, 0.0, 1.0))
    return far + (near - far) * w


def _pigment(rgb: str, transmit: float | None) -> str:
    if transmit is None:
        return f"color rgb {rgb}"
    t = float(np.clip(transmit, 0.0, 1.0))
    return f"color rgbt {rgb[:-1]}, {t:.4g}>"


def _emission_rgb(rgb: tuple[float, float, float], luminosity: float) -> str:
    """Emission RGB; ``luminosity`` 0 → black (no negative / no ambient floor)."""
    lum = max(float(luminosity), 0.0)
    r, g, b = (lum * float(v) for v in rgb)
    return f"<{r:.4g}, {g:.4g}, {b:.4g}>"


def _marker_finish(emission: str) -> str:
    """Regular lit surface with optional additive emission."""
    return (
        "finish { ambient 0 diffuse 0.72 specular 0.14 roughness 0.045 "
        f"emission rgb {emission} }}"
    )


def _ball_light_source(
    center: np.ndarray,
    *,
    light_rgb: tuple[float, float, float],
    intensity: float,
    fade_distance: float,
    fade_power: float = 2.0,
    spotlight: bool = False,
    point_at: np.ndarray | Sequence[float] | None = None,
    spotlight_angle_deg: float = DEFAULT_BALL_SPOTLIGHT_ANGLE_DEG,
    looks_like: str | None = None,
) -> str:
    """POV ``light_source`` (point or camera-aimed spotlight).

    ``looks_like`` is optional POV object text (already indented). Omit it
    for a companion light that must not draw a second sphere.
    """
    c = np.asarray(center, dtype=float).reshape(3)
    intensity = max(float(intensity), 0.0)
    lr, lg, lb = (float(v) * intensity for v in light_rgb)
    fd = max(float(fade_distance), 1e-3)
    fp = max(float(fade_power), 0.0)
    spot = ""
    if spotlight and point_at is not None:
        ang = max(float(spotlight_angle_deg), 1e-3)
        inner = 0.8 * ang
        spot = (
            "  spotlight\n"
            f"  radius {inner:.4g}\n"
            f"  falloff {ang:.4g}\n"
            "  tightness 50\n"
            f"  point_at {pov_vec(point_at)}\n"
        )
    looks = f"  looks_like {{\n{looks_like}  }}\n" if looks_like else ""
    return (
        "light_source {\n"
        f"  {pov_vec(c)}\n"
        f"  color rgb <{lr:.4g}, {lg:.4g}, {lb:.4g}>\n"
        f"{spot}"
        f"  fade_distance {fd:.6g}\n"
        f"  fade_power {fp:.4g}\n"
        f"{looks}"
        "}\n"
    )


def _marker_ball(
    center: np.ndarray,
    *,
    radius: float,
    rgb: str,
    light_rgb: tuple[float, float, float],
    intensity: float,
    emission: str,
    fade_distance: float,
    emit_light: bool,
    transmit: float | None = None,
    hollow: bool = False,
    spotlight: bool = False,
    point_at: np.ndarray | Sequence[float] | None = None,
    spotlight_angle_deg: float = DEFAULT_BALL_SPOTLIGHT_ANGLE_DEG,
    fade_power: float = 2.0,
) -> str:
    """Regular coloured ball with optional additive glow and point/spot light.

    When ``emit_light`` is true, the sphere is a ``looks_like`` on a
    ``light_source`` and is always ``hollow`` so the light can leave.
    ``spotlight`` aims that source at ``point_at`` (the camera) with cone
    ``spotlight_angle_deg``. When ``emit_light`` is false, it is a regular
    ``no_shadow`` sphere; ``hollow`` is used only if a nearby point light
    sits inside it.
    """
    c = np.asarray(center, dtype=float).reshape(3)
    r = float(radius)
    pigment = _pigment(rgb, transmit)
    finish = _marker_finish(emission)
    extra = "no_shadow hollow" if hollow else "no_shadow"
    if not emit_light:
        return sphere(
            c,
            r,
            pigment=pigment,
            finish=finish,
            extra=extra,
        )
    looks_like = (
        f"    sphere {{ 0, {r:.6g}\n"
        f"      pigment {{ {pigment} }}\n"
        f"      {finish}\n"
        "      hollow\n"
        "    }\n"
    )
    return _ball_light_source(
        c,
        light_rgb=light_rgb,
        intensity=intensity,
        fade_distance=fade_distance,
        fade_power=fade_power,
        spotlight=spotlight,
        point_at=point_at,
        spotlight_angle_deg=spotlight_angle_deg,
        looks_like=looks_like,
    )


def _luminous_link(
    a: np.ndarray,
    b: np.ndarray,
    *,
    radius: float,
    pigment: str,
    emission: str,
) -> str:
    """Cylinder from ground truth to a prediction (emissive, slightly transparent)."""
    aa = np.asarray(a, dtype=float).reshape(3)
    bb = np.asarray(b, dtype=float).reshape(3)
    if float(np.linalg.norm(bb - aa)) < 1e-9:
        return ""
    return cylinder(
        aa,
        bb,
        float(radius),
        pigment=f"rgbt {pigment[:-1]}, 0.45>",
        finish=(
            f"finish {{ ambient 0 diffuse 0 phong 0 emission rgb {emission} }}"
        ),
        extra="no_shadow hollow",
    )


# Far enough to frame the bear plus validation markers (mm). Override via
# ``camera_location`` / ``camera_look_at`` from the notebook.
DEFAULT_CAMERA_DISTANCE_SCALE = 3.8
DEFAULT_CAMERA_YAW_DEG = 80.0
DEFAULT_CAMERA_FOV_DEG = 42.0
DEFAULT_CAMERA_ELEV_FRAC = 0.28

FIGURE3_SPLITS = ("train", "validation", "test")
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}


def normalize_figure3_split(split: str) -> str:
    """Map ``train`` / ``validation`` / ``test`` (and common aliases) to canonical names."""
    key = str(split).strip().lower()
    if key not in _SPLIT_ALIASES:
        raise ValueError(
            f"unknown split={split!r}; expected one of {list(FIGURE3_SPLITS)} "
            f"(aliases: training, val, testing)"
        )
    return _SPLIT_ALIASES[key]


def _tracked_meta_split(meta: Mapping[str, Any]) -> str:
    raw = meta.get("split")
    if raw is None or str(raw).strip() == "":
        return "validation"
    return normalize_figure3_split(str(raw))


def figure3_camera_pose(
    mesh: trimesh.Trimesh,
    *,
    camera_location: Sequence[float] | None = None,
    camera_look_at: Sequence[float] | None = None,
    distance_scale: float = DEFAULT_CAMERA_DISTANCE_SCALE,
    yaw_deg: float = DEFAULT_CAMERA_YAW_DEG,
    elev_frac: float = DEFAULT_CAMERA_ELEV_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(location, look_at)`` in simulation millimetres (z-up).

    Defaults use the network-scene azimuth (mostly +X, slight −Y) at a long
    standoff. Either vector may be overridden independently.
    """
    bounds = np.asarray(mesh.bounds, dtype=float)
    center = 0.5 * (bounds[0] + bounds[1])
    span = 0.5 * float(np.max(bounds[1] - bounds[0]))
    look = (
        center
        if camera_look_at is None
        else np.asarray(camera_look_at, dtype=float).reshape(3)
    )
    if camera_location is not None:
        loc = np.asarray(camera_location, dtype=float).reshape(3)
        return loc, look
    dist = max(2.35 * span, 26.0) * float(distance_scale)
    yaw = np.deg2rad(float(yaw_deg))
    loc = np.asarray(look, dtype=float) + np.array(
        [
            dist * np.sin(yaw),
            -dist * np.cos(yaw),
            float(elev_frac) * span,
        ]
    )
    return loc, look


def tracked_sample_ids(
    combined_history: Mapping[str, Any],
    *,
    split: str | None = None,
) -> list[str]:
    """Tracked sample ids, optionally filtered by catalog ``split``."""
    metas = list(combined_history.get("tracked_samples") or [])
    if split is not None:
        want = normalize_figure3_split(split)
        ids = [
            str(s.get("sample_id"))
            for s in metas
            if s.get("sample_id") is not None and _tracked_meta_split(s) == want
        ]
        return [sid for sid in ids if sid]
    ids = [
        str(s.get("sample_id"))
        for s in metas
        if s.get("sample_id") is not None
    ]
    if ids:
        return ids
    seen: list[str] = []
    for rec in combined_history.get("records") or []:
        sid = str(rec.get("sample_id"))
        if sid not in seen:
            seen.append(sid)
    return seen


def figure3_mesh_metrics(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(lo, hi, span)`` in millimetres; ``span`` is half the longest AABB."""
    bounds = np.asarray(mesh.bounds, dtype=float)
    lo, hi = bounds[0], bounds[1]
    span = 0.5 * float(np.max(hi - lo))
    return lo, hi, span


def figure3_mesh_centroid(mesh: trimesh.Trimesh) -> np.ndarray:
    lo, hi, _span = figure3_mesh_metrics(mesh)
    return 0.5 * (lo + hi)


def figure3_apply_bear_placement(
    xyz: Sequence[float] | np.ndarray,
    *,
    mesh: trimesh.Trimesh,
    scale: float = 1.0,
    translate: Sequence[float] | None = None,
) -> np.ndarray:
    """Scale about mesh centroid, then translate (simulation millimetres)."""
    center = figure3_mesh_centroid(mesh)
    offset = (
        np.zeros(3, dtype=float)
        if translate is None
        else np.asarray(translate, dtype=float).reshape(3)
    )
    s = float(scale)
    point = np.asarray(xyz, dtype=float).reshape(3)
    if s == 1.0 and np.allclose(offset, 0.0):
        return point.copy()
    return offset + s * (point - center)


def _figure3_bear_placement_pov(
    *,
    mesh: trimesh.Trimesh,
    scale: float = 1.0,
    translate: Sequence[float] | None = None,
) -> str:
    """POV modifiers: ``translate + scale * (p - centroid)``."""
    center = figure3_mesh_centroid(mesh)
    offset = (
        np.zeros(3, dtype=float)
        if translate is None
        else np.asarray(translate, dtype=float).reshape(3)
    )
    s = float(scale)
    if s == 1.0 and np.allclose(offset, 0.0):
        return ""
    bits = [f"  translate {pov_vec(-center)}\n"]
    if s != 1.0:
        bits.append(f"  scale {s:.6g}\n")
    if not np.allclose(offset, 0.0):
        bits.append(f"  translate {pov_vec(offset)}\n")
    return "".join(bits)


def figure3_epoch_pairs(
    combined_history: Mapping[str, Any] | None,
    epoch: int,
    *,
    split: str | None = None,
    sample_ids: Sequence[str] | None = None,
    record_index: Figure3RecordIndex | None = None,
    split_sample_ids: Sequence[str] | None = None,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Pooling/Fourier records for tracked samples at ``epoch``.

    ``split`` restricts to catalog train/validation/test. ``sample_ids``
    further restricts (or replaces split filtering when provided alone).

    Pass ``record_index`` and ``split_sample_ids`` to avoid rescanning the full
    combined history on every epoch.
    """
    epoch = int(epoch)
    if sample_ids is None:
        if split_sample_ids is not None:
            ids = [str(s) for s in split_sample_ids]
        else:
            if combined_history is None:
                raise ValueError(
                    "combined_history is required when split_sample_ids is not provided"
                )
            ids = tracked_sample_ids(combined_history, split=split)
    else:
        ids = [str(s) for s in sample_ids]
        if split is not None:
            if split_sample_ids is not None:
                allowed = set(split_sample_ids)
            else:
                if combined_history is None:
                    raise ValueError(
                        "combined_history is required when filtering sample_ids by split"
                    )
                allowed = set(tracked_sample_ids(combined_history, split=split))
            ids = [sid for sid in ids if sid in allowed]
    if not ids:
        available = sorted(
            {
                _tracked_meta_split(s)
                for s in (
                    (combined_history or {}).get("tracked_samples") or []
                )
                if s.get("sample_id") is not None
            }
        )
        hint = f" available splits: {available}" if available else ""
        raise ValueError(
            "combined history has no tracked samples"
            + (f" for split={normalize_figure3_split(split)!r}" if split else "")
            + hint
        )
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for sid in ids:
        pooled = records_for_sample(
            combined_history or {},
            sid,
            model_type=MODEL_POOLING,
            record_index=record_index,
        )
        fourier = records_for_sample(
            combined_history or {},
            sid,
            model_type=MODEL_FOURIER,
            record_index=record_index,
        )
        if not pooled or not fourier:
            continue
        pairs.append(
            (sid, _record_at_epoch(pooled, epoch), _record_at_epoch(fourier, epoch))
        )
    if not pairs:
        raise ValueError(f"no pooling/Fourier records at epoch={epoch}")
    return pairs


def figure3_world_pov_text(
    *,
    mesh: trimesh.Trimesh,
    loc: np.ndarray,
    look: np.ndarray,
    fov_deg: float = DEFAULT_CAMERA_FOV_DEG,
    yaw_deg: float = DEFAULT_CAMERA_YAW_DEG,
    light_intensity: float = DEFAULT_LIGHT_INTENSITY,
    with_version: bool = True,
) -> str:
    """Camera, scene key/fill, sky, and floor (no bear, no markers)."""
    lo, hi, _span = figure3_mesh_metrics(mesh)
    light_scale = max(float(light_intensity), 0.0)
    key = 0.34 * light_scale
    fill = 0.12 * light_scale
    z_floor = float(lo[2]) - 0.04 * float(hi[2] - lo[2] + 1e-6)
    bits = [
        comment_block(
            "Figure 3 world: face-on camera (network-scene azimuth, long "
            f"standoff; yaw_deg={float(yaw_deg):g} from −Y toward +X). "
            f"location={pov_vec(loc)} look_at={pov_vec(look)}. "
            "Coordinates = simulation millimetres (identity map). "
            f"light_intensity={light_scale:g} (scene key/fill only)."
        ),
    ]
    if with_version:
        bits.append("#version 3.7;\n")
    bits.extend(
        [
            "global_settings { assumed_gamma 1.0 ambient_light rgb <0.055, 0.055, 0.06> }\n",
            "background { color rgb <0.02, 0.03, 0.06> }\n",
            sky_and_horizon(),
            "camera {\n",
            f"  location {pov_vec(loc)}\n",
            f"  look_at {pov_vec(look)}\n",
            "  sky <0, 0, 1>\n",
            f"  angle {float(fov_deg):.4g}\n",
            "  right x * image_width / image_height\n",
            "}\n",
            f"light_source {{ {pov_vec(loc + np.array([6.0, -4.0, 14.0]))} "
            f"color rgb <{key:.4g}, {key:.4g}, {key * 1.12:.4g}> }}\n",
            f"light_source {{ <-20, -40, 40> color rgb <{fill:.4g}, {fill:.4g}, {fill * 1.2:.4g}> }}\n",
            plane_z(
                z_floor,
                pigment="color rgb <0.22, 0.23, 0.25>",
                finish=(
                    "finish { ambient 0.02 diffuse 0.45 specular 0.15 "
                    "roughness 0.08 reflection 0.04 }"
                ),
            ),
        ]
    )
    return "".join(bits)


def figure3_bear_object_pov_text(
    *,
    mesh: trimesh.Trimesh,
    bear_inc_name: str,
    include_mesh: bool = True,
    scale: float = 1.0,
    translate: Sequence[float] | None = None,
) -> str:
    """Instance the declared ``BearMesh`` plus triangle-edge cylinders.

    When ``include_mesh`` is false the caller is expected to ``#include``
    ``bear_inc_name`` separately (progressive scene files).

    Optional ``scale`` / ``translate`` place a copy without changing catalog
    coordinates (markers use the same placement when attached to this bear).
    """
    _lo, _hi, span = figure3_mesh_metrics(mesh)
    edge_r = max(span * 0.0035, 0.05)
    placement = _figure3_bear_placement_pov(
        mesh=mesh, scale=scale, translate=translate
    )
    bits = [
        comment_block("Bear mesh (same pigment / edges as physical-scene figure)."),
    ]
    if include_mesh:
        bits.append(f'#include "{bear_inc_name}"\n')
    inner = [
        "object {\n",
        "  BearMesh\n",
        "  pigment { color rgbt <0.68, 0.69, 0.71, 0.72> }\n",
        "  finish { ambient 0.02 diffuse 0.82 specular 0 roughness 0.5 reflection 0 }\n",
        "  interior { ior 1.02 }\n",
        "  hollow\n",
        "  double_illuminate\n",
        "}\n",
        bear_triangle_edge_cylinders(mesh, radius=edge_r, transmit=0.88),
    ]
    if placement:
        bits.append("union {\n")
        bits.extend(inner)
        bits.append(placement)
        bits.append("}\n")
    else:
        bits.extend(inner)
    return "".join(bits)


def figure3_markers_pov_text(
    *,
    combined_history: Mapping[str, Any] | None,
    epoch: int,
    mesh: trimesh.Trimesh,
    loc: np.ndarray,
    split: str | None = None,
    bear_id: str = "default",
    bear_scale: float = 1.0,
    bear_translate: Sequence[float] | None = None,
    record_index: Figure3RecordIndex | None = None,
    split_sample_ids: Sequence[str] | None = None,
    ball_lights: bool = False,
    ball_spotlights: bool = DEFAULT_BALL_SPOTLIGHTS,
    ball_spotlight_angle_deg: float = DEFAULT_BALL_SPOTLIGHT_ANGLE_DEG,
    ball_light_intensity: float = DEFAULT_BALL_LIGHT_INTENSITY,
    draw_prediction_links: bool = False,
    distance_luminosity: bool = False,
    distance_transparency: bool = True,
    green_luminosity: float = DEFAULT_GREEN_LUMINOSITY,
    pred_luminosity_at_target: float = DEFAULT_PRED_LUMINOSITY_AT_TARGET,
    pooling_luminosity_at_target: float | None = None,
    fourier_luminosity_at_target: float | None = None,
    pred_luminosity_hold_mm: float = DEFAULT_PRED_HOLD_MM,
    pred_luminosity_zero_mm: float = DEFAULT_PRED_APPROACH_ZERO_MM,
    pred_transmit_hold_mm: float = DEFAULT_PRED_HOLD_MM,
    pred_transmit_far: float = DEFAULT_PRED_TRANSMIT_FAR,
    pred_transmit_zero_mm: float = DEFAULT_PRED_APPROACH_ZERO_MM,
    pred_transmit_at_target: float = DEFAULT_PRED_TRANSMIT_AT_TARGET,
) -> str:
    """Green targets + red/blue predictions (and optional ball lights) for one epoch.

    ``split`` selects train/validation/test samples; ``bear_id`` is recorded in
    the include so several bears can each carry their own marker layer.
    """
    pairs = figure3_epoch_pairs(
        combined_history,
        epoch,
        split=split,
        record_index=record_index,
        split_sample_ids=split_sample_ids,
    )
    _lo, _hi, span = figure3_mesh_metrics(mesh)
    placement_scale = max(float(bear_scale), 1e-9)
    ball_r = max(0.022 * span * placement_scale, 0.32 * placement_scale)
    link_r = 0.28 * ball_r

    def _placed_xyz(raw: Sequence[float]) -> np.ndarray:
        return simulation_xyz_to_pov(
            figure3_apply_bear_placement(
                raw,
                mesh=mesh,
                scale=placement_scale,
                translate=bear_translate,
            )
        )
    mean_p = float(np.mean([float(p["localization_error"]) for _s, p, _f in pairs]))
    mean_f = float(np.mean([float(f["localization_error"]) for _s, _p, f in pairs]))
    bits = [
        comment_block(
            "Figure 3: M8 pooling (red) vs Fourier pooling (blue) localization "
            "convergence. "
            f"bear_id={bear_id} split="
            f"{normalize_figure3_split(split) if split is not None else 'all'}. "
            "Tracked particles: green GT linked to current "
            "current red/blue predictions. Marker spheres are self-emissive even "
            "when per-ball point lights are off, but at a reduced baseline so they "
            "do not wash out the bear. Optional distance-based luminosity can "
            "scale that emission with target approach. "
            f"ball_lights={bool(ball_lights)} "
            f"ball_spotlights={bool(ball_spotlights)} "
            f"ball_spotlight_angle_deg={float(ball_spotlight_angle_deg):g} "
            f"ball_light_intensity={max(float(ball_light_intensity), 0.0):g} "
            f"draw_prediction_links={bool(draw_prediction_links)} "
            f"distance_luminosity={bool(distance_luminosity)} "
            f"distance_transparency={bool(distance_transparency)} "
            f"green_luminosity={float(green_luminosity):g} "
            f"pred_luminosity_at_target={float(pred_luminosity_at_target):g} "
            f"pooling_luminosity_at_target="
            f"{float(pooling_luminosity_at_target if pooling_luminosity_at_target is not None else pred_luminosity_at_target):g} "
            f"fourier_luminosity_at_target="
            f"{float(fourier_luminosity_at_target if fourier_luminosity_at_target is not None else pred_luminosity_at_target):g} "
            f"pred_luminosity_hold_mm={float(pred_luminosity_hold_mm):g} "
            f"pred_luminosity_zero_mm={float(pred_luminosity_zero_mm):g} "
            f"pred_transmit_hold_mm={float(pred_transmit_hold_mm):g} "
            f"pred_transmit_far={float(pred_transmit_far):g} "
            f"pred_transmit_zero_mm={float(pred_transmit_zero_mm):g} "
            f"pred_transmit_at_target={float(pred_transmit_at_target):g}. "
            f"epoch={int(epoch)} n_val={len(pairs)} "
            f"mean_pooled_err={mean_p:.4g} mean_fourier_err={mean_f:.4g}"
        ),
        comment_block(
            "Validation particles: regular lit balls "
            + (
                "+ predicted-ball lights only inside "
                f"pred_luminosity_hold_mm={float(pred_luminosity_hold_mm):g} mm; "
                "a second green spotlight is added at the prediction and the "
                "target is hollow when the prediction centre is inside the "
                "green radius"
                + (
                    f"; camera-aimed spotlights, cone {float(ball_spotlight_angle_deg):g} deg. "
                    if ball_spotlights
                    else "; omnidirectional point lights. "
                )
                if ball_lights
                else "(no per-ball light_source). "
            )
            + (
                "Target→pred cylinders drawn. "
                if draw_prediction_links
                else "No target→pred cylinders. "
            )
            + (
                "Emission scales with distance to green target. "
                if distance_luminosity
                else "Constant ball emission. "
            )
            + (
                "Predicted balls fade when far (rgbt transmit). "
                if distance_transparency
                else "Predicted balls opaque. "
            )
        ),
    ]
    fade = max(6.5, 0.55 * span)
    ball_fade = max(1.5 * span, 12.0) if bool(ball_spotlights) else fade
    ball_fade_power = 1.0 if bool(ball_spotlights) else 2.0
    pred_r = 0.92 * ball_r
    use_ball_lights = bool(ball_lights)
    use_spot = bool(ball_spotlights)
    spot_ang = float(ball_spotlight_angle_deg)
    ball_i_scale = max(float(ball_light_intensity), 0.0)
    use_links = bool(draw_prediction_links)
    use_lum = bool(distance_luminosity)
    use_xmit = bool(distance_transparency)
    lum_hold_mm = max(float(pred_luminosity_hold_mm), 0.0) * placement_scale
    zero_mm = max(float(pred_luminosity_zero_mm), 1e-9) * placement_scale
    xmit_hold_mm = max(float(pred_transmit_hold_mm), 0.0) * placement_scale
    xmit_zero_mm = max(float(pred_transmit_zero_mm), 1e-9) * placement_scale
    at_tgt = max(float(pred_luminosity_at_target), 0.0)
    pool_at_tgt = max(
        float(at_tgt if pooling_luminosity_at_target is None else pooling_luminosity_at_target),
        0.0,
    )
    four_at_tgt = max(
        float(at_tgt if fourier_luminosity_at_target is None else fourier_luminosity_at_target),
        0.0,
    )
    xmit_far = float(pred_transmit_far)
    xmit_near = float(pred_transmit_at_target)
    green_base = max(float(green_luminosity), 0.0)
    pred_pool_base = pool_at_tgt
    pred_four_base = four_at_tgt
    for sid, p_now, f_now in pairs:
        y_true = _placed_xyz(p_now["y_true"])
        y_p = _placed_xyz(p_now["y_pred"])
        y_f = _placed_xyz(f_now["y_pred"])
        d_p = float(np.linalg.norm(y_p - y_true))
        d_f = float(np.linalg.norm(y_f - y_true))
        w_p = approach_weight(d_p, hold_mm=lum_hold_mm, zero_mm=zero_mm)
        w_f = approach_weight(d_f, hold_mm=lum_hold_mm, zero_mm=zero_mm)
        w_g = max(w_p, w_f)
        lum_g = green_base * (w_g if use_lum else 1.0)
        lum_p = pred_pool_base * (w_p if use_lum else 1.0)
        lum_f = pred_four_base * (w_f if use_lum else 1.0)
        xmit_p = (
            distance_transmit(
                d_p,
                hold_mm=xmit_hold_mm,
                zero_mm=xmit_zero_mm,
                transmit_far=xmit_far,
                transmit_at_target=xmit_near,
            )
            if use_xmit
            else None
        )
        xmit_f = (
            distance_transmit(
                d_f,
                hold_mm=xmit_hold_mm,
                zero_mm=xmit_zero_mm,
                transmit_far=xmit_far,
                transmit_at_target=xmit_near,
            )
            if use_xmit
            else None
        )
        i_p = lum_p * LIGHT_I_MAX * ball_i_scale
        i_f = lum_f * LIGHT_I_MAX * ball_i_scale
        i_g = lum_g * LIGHT_I_MAX * ball_i_scale
        light_p = use_ball_lights and d_p <= lum_hold_mm
        light_f = use_ball_lights and d_f <= lum_hold_mm
        inside_p = d_p <= ball_r
        inside_f = d_f <= ball_r
        green_spot_p = light_p and inside_p
        green_spot_f = light_f and inside_f
        green_open = green_spot_p or green_spot_f
        bits.append(
            comment_block(
                f"sample {sid}  d_pool={d_p:.3g} d_fourier={d_f:.3g} "
                f"w_pool={w_p:.3f} w_fourier={w_f:.3f}  "
                f"xmit_pool={xmit_p if xmit_p is not None else 'n/a'} "
                f"xmit_fourier={xmit_f if xmit_f is not None else 'n/a'}  "
                f"light_pool={light_p} light_fourier={light_f} "
                f"inside_pool={inside_p} inside_fourier={inside_f} "
                f"green_spot_pool={green_spot_p} green_spot_fourier={green_spot_f} "
                f"green_hollow={green_open}"
            )
        )
        bits.append(
            _marker_ball(
                y_true,
                radius=ball_r,
                rgb=TARGET_RGB,
                light_rgb=TARGET_LIGHT,
                intensity=i_g,
                emission=_emission_rgb(TARGET_LIGHT, lum_g),
                fade_distance=fade,
                emit_light=False,
                transmit=DEFAULT_TARGET_HIT_TRANSMIT if green_open else None,
                hollow=green_open,
            )
        )
        if use_links:
            bits.append(
                _luminous_link(
                    y_true,
                    y_p,
                    radius=link_r,
                    pigment=POOLING_RGB,
                    emission=_emission_rgb(POOLING_LIGHT, lum_p),
                )
            )
        bits.append(
            _marker_ball(
                y_p,
                radius=pred_r,
                rgb=POOLING_RGB,
                light_rgb=POOLING_LIGHT,
                intensity=i_p,
                emission=_emission_rgb(POOLING_LIGHT, lum_p),
                fade_distance=ball_fade,
                fade_power=ball_fade_power,
                emit_light=light_p,
                transmit=xmit_p,
                spotlight=use_spot,
                point_at=loc,
                spotlight_angle_deg=spot_ang,
            )
        )
        if green_spot_p:
            bits.append(
                _ball_light_source(
                    y_p,
                    light_rgb=TARGET_LIGHT,
                    intensity=i_p,
                    fade_distance=ball_fade,
                    fade_power=ball_fade_power,
                    spotlight=use_spot,
                    point_at=loc,
                    spotlight_angle_deg=spot_ang,
                )
            )
        if use_links:
            bits.append(
                _luminous_link(
                    y_true,
                    y_f,
                    radius=link_r,
                    pigment=FOURIER_RGB,
                    emission=_emission_rgb(FOURIER_LIGHT, lum_f),
                )
            )
        bits.append(
            _marker_ball(
                y_f,
                radius=pred_r,
                rgb=FOURIER_RGB,
                light_rgb=FOURIER_LIGHT,
                intensity=i_f,
                emission=_emission_rgb(FOURIER_LIGHT, lum_f),
                fade_distance=ball_fade,
                fade_power=ball_fade_power,
                emit_light=light_f,
                transmit=xmit_f,
                spotlight=use_spot,
                point_at=loc,
                spotlight_angle_deg=spot_ang,
            )
        )
        if green_spot_f:
            bits.append(
                _ball_light_source(
                    y_f,
                    light_rgb=TARGET_LIGHT,
                    intensity=i_f,
                    fade_distance=ball_fade,
                    fade_power=ball_fade_power,
                    spotlight=use_spot,
                    point_at=loc,
                    spotlight_angle_deg=spot_ang,
                )
            )
    return "".join(bits)


def build_figure3_epoch_pov(
    *,
    combined_history: Mapping[str, Any],
    epoch: int,
    bear_inc_name: str,
    mesh: trimesh.Trimesh,
    sample_id: str | None = None,
    fov_deg: float = DEFAULT_CAMERA_FOV_DEG,
    distance_scale: float = DEFAULT_CAMERA_DISTANCE_SCALE,
    yaw_deg: float = DEFAULT_CAMERA_YAW_DEG,
    camera_location: Sequence[float] | None = None,
    camera_look_at: Sequence[float] | None = None,
    light_intensity: float = DEFAULT_LIGHT_INTENSITY,
    ball_lights: bool = False,
    ball_spotlights: bool = DEFAULT_BALL_SPOTLIGHTS,
    ball_spotlight_angle_deg: float = DEFAULT_BALL_SPOTLIGHT_ANGLE_DEG,
    ball_light_intensity: float = DEFAULT_BALL_LIGHT_INTENSITY,
    draw_prediction_links: bool = False,
    distance_luminosity: bool = False,
    distance_transparency: bool = True,
    green_luminosity: float = DEFAULT_GREEN_LUMINOSITY,
    pred_luminosity_at_target: float = DEFAULT_PRED_LUMINOSITY_AT_TARGET,
    pooling_luminosity_at_target: float | None = None,
    fourier_luminosity_at_target: float | None = None,
    pred_luminosity_hold_mm: float = DEFAULT_PRED_HOLD_MM,
    pred_luminosity_zero_mm: float = DEFAULT_PRED_APPROACH_ZERO_MM,
    pred_transmit_hold_mm: float = DEFAULT_PRED_HOLD_MM,
    pred_transmit_far: float = DEFAULT_PRED_TRANSMIT_FAR,
    pred_transmit_zero_mm: float = DEFAULT_PRED_APPROACH_ZERO_MM,
    pred_transmit_at_target: float = DEFAULT_PRED_TRANSMIT_AT_TARGET,
) -> str:
    """One frame: face-on bear; all validation GTs (green) linked to red/blue preds.

    ``sample_id`` is accepted for back-compat but ignored: every tracked
    validation sample in the JSON is drawn. Camera defaults to the
    network-scene azimuth at a long standoff; pass ``camera_location`` /
    ``camera_look_at`` (mm) to override. ``light_intensity`` scales the scene
    key/fill lights only (1 = default).
    ``ball_lights`` (default False) adds a faded ``light_source`` only on
    predicted balls that are within ``pred_luminosity_hold_mm`` of the green
    target. If the prediction centre also lies inside the green ball, a
    second green spotlight is added at the prediction (same cone) while the
    red/blue light stays; the target is made hollow/transmissive so both
    can escape. Green targets never carry their own ``light_source``.
    ``ball_spotlights`` (default True) makes those sources POV ``spotlight``s
    aimed at the camera, with cone ``ball_spotlight_angle_deg`` (default 15°).
    ``ball_light_intensity`` scales only those ``light_source`` colours
    (default 8); marker emission and the scene key/fill are unchanged.
    The bear mesh is ``hollow`` + ``double_illuminate`` so interior cones
    shade the camera-facing skin as a local patch (no volumetric shaft).
    ``draw_prediction_links`` draws target→pred cylinders (default False).
    ``distance_luminosity`` scales emission with distance (default False).
    ``distance_transparency`` fades predicted balls when far (default True):
    each effect holds its near-target value through its ``*_hold_mm`` radius,
    then decays linearly to its background value at its ``*_zero_mm`` radius.
    """
    del sample_id  # unused; all tracked samples are rendered
    loc, look = figure3_camera_pose(
        mesh,
        camera_location=camera_location,
        camera_look_at=camera_look_at,
        distance_scale=distance_scale,
        yaw_deg=yaw_deg,
    )
    return (
        figure3_world_pov_text(
            mesh=mesh,
            loc=loc,
            look=look,
            fov_deg=fov_deg,
            yaw_deg=yaw_deg,
            light_intensity=light_intensity,
            with_version=True,
        )
        + figure3_bear_object_pov_text(mesh=mesh, bear_inc_name=bear_inc_name)
        + figure3_markers_pov_text(
            combined_history=combined_history,
            epoch=epoch,
            mesh=mesh,
            loc=loc,
            ball_lights=ball_lights,
            ball_spotlights=ball_spotlights,
            ball_spotlight_angle_deg=ball_spotlight_angle_deg,
            ball_light_intensity=ball_light_intensity,
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
    )


def prepare_bear_inc(
    stl_path: Path,
    inc_path: Path,
) -> tuple[str, trimesh.Trimesh]:
    mesh = trimesh.load(stl_path, force="mesh")
    write_stl_mesh2_inc(stl_path, inc_path)
    return inc_path.name, mesh
