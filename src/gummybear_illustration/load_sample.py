"""Load M8 physical-setup fields from a sequence ``manifest.json``.

Geometry markers come from the manifest and STL. Ray polylines are not
stored; :func:`illustration_optical_ray_segments` rebuilds a downsampled
``make_source_ray_bundle`` + ``refract_ray_bundle`` +
``in_object_segments_from_rays`` (the same optical entries as generation).
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gummybear.datasets.generation_plan import DEFAULT_RUNTIME_SETTINGS
from gummybear.geometry.io import load_stl
from gummybear.optics.light_source import PointLightConfig
from gummybear.optics.material import OpticalMaterialConfig
from gummybear.optics.refraction import refract_ray_bundle
from gummybear.optics.source_deposition import in_object_segments_from_rays
from gummybear.optics.source_sampling import SourceSamplingParams, make_source_ray_bundle
from gummybear.paths import display_path
from gummybear.rays.visibility import first_visible_hits_with_points

from .paths import default_m8_data_root, repo_root

# Illustration downsample of ``SourceSamplingParams.n_rays`` (generation uses
# workbook ``num_source_rays``, often hundreds). Same sampler, fewer cylinders.
ILLUSTRATION_N_SOURCE_RAYS = 96
ILLUSTRATION_SOURCE_SEED = 0


def _warn_portable(message: str, *, stacklevel: int = 2) -> None:
    """Emit ``UserWarning`` with a display-safe filename (no absolute root)."""
    frame = sys._getframe(stacklevel)
    warnings.warn_explicit(
        message,
        UserWarning,
        display_path(frame.f_code.co_filename),
        int(frame.f_lineno),
    )


def _as_xyz(value: Any, *, field_name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a length-3 list; got {value!r}")
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must be finite; got {value!r}")
    return arr


def _vec3(x: float, y: float, z: float) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=float)


@dataclass(frozen=True)
class PhysicalSetup:
    """World-space markers for one M8 sequence / one acquisition view.

    Attributes:
        sequence_id: Manifest sequence identifier.
        manifest_path: Path to ``manifest.json``.
        stl_path: Resolved STL used by the sample.
        particle_center: Sphere centre (mm), from particle setup metadata.
        particle_radius: Sphere radius (mm), from particle setup metadata.
        light_position: Point-light position (mm), from optical setup metadata.
        camera_position: Acquisition pinhole (mm).
        camera_look_at: Look-at point (mm).
        camera_up: Up hint for the acquisition camera.
        camera_fov_deg: Square pinhole FOV (degrees).
        camera_angle_deg: Selected orbit angle.
        refractive_index: Phantom IOR when present on the optical setup.
        illumination_rays: Exterior source segments (light → first hit), mm.
        illumination_rays_are_fallback: True when segments are not from a ray dump.
        refracted_rays: In-object chords after Snell entry (mm).
        warnings: Human-readable notices already emitted or to display.
        frame_anomaly_raw: ``(angle_deg, anomaly.raw.tif path)`` for frames
            that have an anomaly sidecar.
    """

    sequence_id: str
    manifest_path: Path
    stl_path: Path
    particle_center: np.ndarray
    particle_radius: float
    light_position: np.ndarray
    camera_position: np.ndarray
    camera_look_at: np.ndarray
    camera_up: np.ndarray
    camera_fov_deg: float
    camera_angle_deg: float
    refractive_index: float | None
    illumination_rays: tuple[tuple[np.ndarray, np.ndarray], ...]
    illumination_rays_are_fallback: bool
    refracted_rays: tuple[tuple[np.ndarray, np.ndarray], ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    frame_anomaly_raw: tuple[tuple[float, Path], ...] = field(default_factory=tuple)


def resolve_stl_path(
    stl_path: str | Path,
    *,
    repo: Path,
    manifest_path: Path,
) -> Path:
    """Resolve a portable (usually repo-relative) STL path from the manifest."""
    raw = Path(stl_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                repo / raw,
                manifest_path.parent / raw,
                repo / "cad" / raw.name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"STL not found for phantom path {stl_path!r} "
        f"(manifest={manifest_path})"
    )


def _particle_from_setups(setups: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    """Particle centre/radius from ``setups.particles.items`` or ``setups.particle``.

    These are workbook scientific coordinates copied into the manifest, not
    estimated from images.
    """
    particles_block = setups.get("particles")
    item: Mapping[str, Any] | None = None
    if isinstance(particles_block, Mapping):
        items = particles_block.get("items")
        if isinstance(items, list) and items:
            if not isinstance(items[0], Mapping):
                raise ValueError("setups.particles.items[0] must be an object")
            item = items[0]
    if item is None:
        particle = setups.get("particle")
        if isinstance(particle, Mapping) and "center_x" in particle:
            item = particle
    if item is None:
        raise ValueError(
            "Manifest setups are missing particle centre/radius "
            "(expected setups.particles.items[0] or setups.particle)."
        )
    center = _vec3(item["center_x"], item["center_y"], item["center_z"])
    radius = float(item["radius"])
    if radius <= 0.0:
        raise ValueError(f"particle radius must be positive; got {radius}")
    return center, radius


def _point_light_from_setups(setups: Mapping[str, Any]) -> PointLightConfig:
    """``PointLightConfig`` from ``setups.optical`` (same fields as generation)."""
    position = _light_from_setups(setups)
    optical = setups["optical"]
    kwargs: dict[str, Any] = {
        "position": (float(position[0]), float(position[1]), float(position[2]))
    }
    if "source_intensity" in optical:
        kwargs["intensity"] = float(optical["source_intensity"])
    if "source_falloff" in optical:
        kwargs["falloff"] = str(optical["source_falloff"])
    return PointLightConfig(**kwargs)


def _light_from_setups(setups: Mapping[str, Any]) -> np.ndarray:
    """Point-light position from ``setups.optical`` workbook fields."""
    optical = setups.get("optical")
    if not isinstance(optical, Mapping):
        raise ValueError("Manifest setups.optical is missing.")
    try:
        return _vec3(
            optical["light_position_x"],
            optical["light_position_y"],
            optical["light_position_z"],
        )
    except KeyError as exc:
        raise ValueError(
            "setups.optical is missing light_position_x/y/z "
            "(these are scientific metadata, not visualization defaults)."
        ) from exc


def _frame_anomaly_raw_paths(
    frames: Sequence[Mapping[str, Any]],
    sequence_dir: Path,
) -> tuple[tuple[float, Path], ...]:
    """Collect existing ``anomaly_raw`` (else ``anomaly``) files per frame angle."""
    found: list[tuple[float, Path]] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        names = frame.get("filenames")
        if not isinstance(names, Mapping):
            continue
        rel = names.get("anomaly_raw") or names.get("anomaly")
        if not rel:
            continue
        path = sequence_dir / str(rel)
        if path.is_file():
            found.append((float(frame["angle_deg"]), path))
    return tuple(found)


def _select_frame(
    frames: Sequence[Mapping[str, Any]],
    camera_angle_deg: float,
    *,
    atol_deg: float = 0.51,
) -> Mapping[str, Any]:
    if not frames:
        raise ValueError("Manifest has no frames list.")
    matches = [
        f
        for f in frames
        if abs(float(f["angle_deg"]) - float(camera_angle_deg)) <= atol_deg
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple frames match angle_deg≈{camera_angle_deg}"
        )
    raise ValueError(
        f"No frame with angle_deg≈{camera_angle_deg} "
        f"(have {[float(f.get('angle_deg')) for f in frames]})"
    )


def reconstruct_orbit_camera(
    frame: Mapping[str, Any],
    *,
    look_at: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct pinhole pose from orbit metadata (same geometry as generation).

    Mirrors ``gummybear.datasets.sequence_generation._orbit_camera``:
    reference direction in the plane ⟂ axis is ``[0,-1,0]``, then rotate by
    ``angle_deg`` and elevate by ``elevation_deg``.

    This is **reconstruction of the real acquisition pose**, used only when
    ``camera_position`` is absent from the frame dict. It is not an invented
    viewpoint.
    """
    axis = np.asarray(frame.get("axis", (0.0, 0.0, 1.0)), dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("Camera orbit axis must be nonzero.")
    axis = axis / axis_norm
    reference = np.array([0.0, -1.0, 0.0], dtype=float)
    reference = reference - float(np.dot(reference, axis)) * axis
    if float(np.linalg.norm(reference)) < 1e-8:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
        reference = reference - float(np.dot(reference, axis)) * axis
    reference = reference / np.linalg.norm(reference)
    tangent = np.cross(axis, reference)
    angle = math.radians(float(frame["angle_deg"]))
    elevation = math.radians(float(frame.get("elevation_deg", 0.0)))
    radial = np.cos(angle) * reference + np.sin(angle) * tangent
    distance = float(frame["distance"])
    offset = distance * (np.cos(elevation) * radial + np.sin(elevation) * axis)
    camera_position = look_at + offset
    up = axis.copy()
    view = look_at - camera_position
    view_n = view / np.linalg.norm(view)
    if abs(float(np.dot(view_n, up))) > 0.98:
        up = tangent
    return camera_position, up


def load_illumination_ray_segments(
    payload: Mapping[str, Any],
    sequence_dir: Path,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], bool]:
    """Return real ray segments if a sidecar exists; else empty + fallback flag.

    Expected optional sidecar (not written by current generation):
    ``sequence_dir / "illustration_source_rays.json"`` with
    ``{"segments": [[[x,y,z],[x,y,z]], ...]}``.

    The production manifest only stores ray *counts* under
    ``generation.diagnostics``; it does not store polylines. This hook is the
    extension point for consuming real segments later without changing the
    optical pipeline.
    """
    sidecar = sequence_dir / "illustration_source_rays.json"
    if sidecar.is_file():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        raw = data.get("segments")
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{sidecar} has no segments list")
        segs = []
        for pair in raw:
            origin = _as_xyz(pair[0], field_name="segment.origin")
            end = _as_xyz(pair[1], field_name="segment.end")
            segs.append((origin, end))
        return tuple(segs), False
    # Manifest diagnostics are counts only — not geometry.
    _ = payload.get("generation", {})
    return (), True


def illustration_optical_ray_segments(
    *,
    light: PointLightConfig,
    mesh,
    n_to: float | None,
    n_from: float | None = None,
    n_rays: int = ILLUSTRATION_N_SOURCE_RAYS,
    seed: int | None = ILLUSTRATION_SOURCE_SEED,
) -> tuple[
    tuple[tuple[np.ndarray, np.ndarray], ...],
    tuple[tuple[np.ndarray, np.ndarray], ...],
]:
    """Downsampled generation optical path: source bundle → Snell entry → in-object chords.

    Same callables as ``sequence_generation.compute_clean``:
    ``make_source_ray_bundle``, ``refract_ray_bundle``, ``in_object_segments_from_rays``.
    ``n_rays`` / ``seed`` are illustration-only (generation uses workbook counts
    and ``seed=None``). ``n_from`` defaults to workbook runtime air index.
    """
    if n_from is None:
        n_from = float(DEFAULT_RUNTIME_SETTINGS["n_from"])
    if n_to is None:
        n_to = float(OpticalMaterialConfig().n_refractive)
    bbox = np.asarray(mesh.bounds, dtype=float)
    bundle = make_source_ray_bundle(
        light,
        bbox,
        SourceSamplingParams(mode="point_uniform", n_rays=int(n_rays), seed=seed),
    )
    hit_valid, _depth, hit_faces, hit_points = first_visible_hits_with_points(
        mesh, bundle
    )
    illumination = []
    for i in range(len(bundle.origins)):
        if not bool(hit_valid[i]) or int(hit_faces[i]) < 0:
            continue
        end = np.asarray(hit_points[i], dtype=float)
        if not np.all(np.isfinite(end)):
            continue
        illumination.append((np.asarray(bundle.origins[i], dtype=float), end))
    if not illumination:
        raise ValueError(
            "make_source_ray_bundle produced no mesh hits to draw "
            "(check light position vs STL)."
        )
    refracted = refract_ray_bundle(mesh, bundle, n_from=float(n_from), n_to=float(n_to))
    interior = in_object_segments_from_rays(
        mesh,
        refracted.rays,
        parent_ray_ids=refracted.parent_indices,
    )
    refracted_segs = tuple(
        (np.asarray(s, dtype=float), np.asarray(e, dtype=float))
        for s, e in zip(interior.starts, interior.ends)
    )
    return tuple(illumination), refracted_segs


def fallback_illumination_rays(
    *,
    light: PointLightConfig,
    mesh,
    n_rays: int = ILLUSTRATION_N_SOURCE_RAYS,
    seed: int | None = ILLUSTRATION_SOURCE_SEED,
    n_to: float | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Exterior source segments only; see :func:`illustration_optical_ray_segments`."""
    illumination, _refracted = illustration_optical_ray_segments(
        light=light, mesh=mesh, n_to=n_to, n_rays=n_rays, seed=seed
    )
    return illumination


def discover_m8_manifests(data_root: Path) -> list[Path]:
    """Sorted ``manifest.json`` paths one level under ``data_root``."""
    if not data_root.is_dir():
        return []
    found = sorted(data_root.glob("*/manifest.json"))
    return [p.resolve() for p in found if p.is_file()]


def load_m8_physical_setup(
    *,
    sample_index: int = 0,
    camera_angle_deg: float = 180.0,
    manifest_path: str | Path | None = None,
    data_root: str | Path | None = None,
    repo_root_path: str | Path | None = None,
    n_illustration_rays: int | None = None,
) -> PhysicalSetup:
    """Load particle, light, camera, and STL for one M8 sequence.

    ``sample_index`` indexes discovered manifests (lexicographic sequence
    directory names) when ``manifest_path`` is omitted.
    ``n_illustration_rays`` is the illustration downsample of source rays
    (``None`` = package default, ``0`` = light marker only, no cylinders).
    """
    repo = repo_root(repo_root_path) if repo_root_path is not None else repo_root()
    notices: list[str] = []
    if manifest_path is not None:
        man_path = Path(manifest_path).expanduser().resolve()
    else:
        root = Path(data_root) if data_root is not None else default_m8_data_root(repo)
        manifests = discover_m8_manifests(root)
        if not manifests:
            raise FileNotFoundError(
                f"No M8 manifests under {root}. Generate the M8 corpus or "
                "pass manifest_path= explicitly."
            )
        if sample_index < 0 or sample_index >= len(manifests):
            raise IndexError(
                f"sample_index={sample_index} out of range "
                f"(n_manifests={len(manifests)})"
            )
        man_path = manifests[int(sample_index)]
    if not man_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {man_path}")

    payload = json.loads(man_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest is not a JSON object: {man_path}")
    sequence_id = str(payload.get("sequence_id") or man_path.parent.name)
    phantom = payload.get("phantom")
    if not isinstance(phantom, Mapping) or "stl_path" not in phantom:
        raise ValueError("Manifest phantom.stl_path is required.")
    stl_path = resolve_stl_path(str(phantom["stl_path"]), repo=repo, manifest_path=man_path)

    setups = payload.get("setups")
    if not isinstance(setups, Mapping):
        raise ValueError("Manifest setups object is required.")
    particle_center, particle_radius = _particle_from_setups(setups)
    light = _point_light_from_setups(setups)
    light_position = np.asarray(light.position, dtype=float)
    optical = setups["optical"] if isinstance(setups.get("optical"), Mapping) else {}
    ior = optical.get("refractive_index")
    refractive_index = float(ior) if ior is not None else None
    surface_mesh = load_stl(stl_path)

    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Manifest frames list is required.")
    frame = _select_frame(frames, camera_angle_deg)

    look_at_raw = frame.get("look_at")
    if look_at_raw is not None:
        look_at = _as_xyz(look_at_raw, field_name="frame.look_at")
    else:
        look_at = np.asarray(surface_mesh.bounds, dtype=float).mean(axis=0)
        notices.append(
            "frame.look_at missing; using mesh AABB centroid "
            "(CAMERA_LOOK_AT_POLICY=mesh_bounds_centroid)."
        )

    if frame.get("camera_position") is not None:
        camera_position = _as_xyz(frame["camera_position"], field_name="frame.camera_position")
        up_raw = frame.get("up", [0.0, 0.0, 1.0])
        camera_up = _as_xyz(up_raw, field_name="frame.up")
    else:
        if "distance" not in frame:
            raise ValueError(
                "Frame has neither camera_position nor distance; "
                "cannot reconstruct the acquisition pose."
            )
        camera_position, camera_up = reconstruct_orbit_camera(frame, look_at=look_at)
        notices.append(
            "frame.camera_position missing; reconstructed with the generation "
            "orbit formula from angle/distance/elevation (not an invented pose)."
        )

    fov = float(frame.get("fov_deg", 35.0))
    if "fov_deg" not in frame:
        notices.append("frame.fov_deg missing; using 35° (generation default).")

    n_draw = (
        ILLUSTRATION_N_SOURCE_RAYS
        if n_illustration_rays is None
        else int(n_illustration_rays)
    )
    if n_draw < 0:
        raise ValueError(f"n_illustration_rays must be >= 0, got {n_draw}")

    rays, need_fallback = load_illumination_ray_segments(payload, man_path.parent)
    optical_refracted: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
    if n_draw == 0:
        rays = ()
        need_fallback = False
    else:
        optical_illum, optical_refracted = illustration_optical_ray_segments(
            light=light,
            mesh=surface_mesh,
            n_to=refractive_index,
            n_rays=n_draw,
        )
        if need_fallback:
            rays = optical_illum
            msg = (
                "Ray polylines are not in the manifest; drawing a downsampled "
                "FALLBACK from make_source_ray_bundle + refract_ray_bundle + "
                "in_object_segments_from_rays (point_uniform, "
                f"n_rays={n_draw}, seed={ILLUSTRATION_SOURCE_SEED})."
            )
            notices.append(msg)
            _warn_portable(msg)

    for note in notices:
        if "FALLBACK" not in note and "reconstructed" in note:
            _warn_portable(note)

    return PhysicalSetup(
        sequence_id=sequence_id,
        manifest_path=man_path,
        stl_path=stl_path,
        particle_center=particle_center,
        particle_radius=particle_radius,
        light_position=light_position,
        camera_position=camera_position,
        camera_look_at=look_at,
        camera_up=camera_up,
        camera_fov_deg=fov,
        camera_angle_deg=float(frame["angle_deg"]),
        refractive_index=refractive_index,
        illumination_rays=rays,
        illumination_rays_are_fallback=need_fallback,
        refracted_rays=optical_refracted,
        warnings=tuple(notices),
        frame_anomaly_raw=_frame_anomaly_raw_paths(frames, man_path.parent),
    )
