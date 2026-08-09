"""Deterministic SHA256 cache-key construction for shared generation state.

Builds canonical JSON file payloads and digests for clean optical source, particle
source, camera visibility, and fluence field ``Phi`` sampling localization. Diffusion
finite-element method (FEM) settings and display JPEG preview options are provenance
only and must not appear on these keys.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from gummybear import __version__ as package_version

CLEAN_OPTICAL_ALGORITHM_VERSION = "m6-clean-optical-v4"
PARTICLE_SOURCE_ALGORITHM_VERSION = "m6-particle-source-v4"
CAMERA_VISIBILITY_ALGORITHM_VERSION = "m6-camera-visibility-v1"
PHI_SAMPLING_LOCALIZATION_ALGORITHM_VERSION = "m6-phi-sampling-localization-v1"
SURFACE_MESH_PROCESSING_VERSION = "m6-surface-mesh-v1"
DIFFUSION_MESH_GENERATION_IDENTITY = "m6-netgen-target-elements-1000-v1"
CAMERA_LOOK_AT_POLICY = "mesh_bounds_centroid"
DEFAULT_BARYCENTRIC_TOLERANCE = 1e-10


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a mapping to stable UTF-8 JSON file bytes."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def hash_cache_payload(payload: Mapping[str, Any]) -> str:
    """Return the SHA256 hex digest of a canonicalized cache-key payload."""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def short_cache_id(full_digest: str, *, n: int = 16) -> str:
    """Return a filesystem-safe short prefix of a full digest."""
    if n < 8 or n > len(full_digest):
        raise ValueError(f"Invalid short-id length: {n}")
    return full_digest[:n]


def _normalize_float(value: Any) -> float:
    return float(value)


def _normalize_int(value: Any) -> int:
    return int(value)


def _normalize_str(value: Any) -> str:
    return str(value)


def clean_optical_cache_key_payload(
    *,
    stl_sha256: str | None = None,
    stl_path: str | None = None,
    surface_mesh_processing_version: str = SURFACE_MESH_PROCESSING_VERSION,
    diffusion_mesh_generation_identity: str = DIFFUSION_MESH_GENERATION_IDENTITY,
    illumination_kind: str,
    light_position_x: float,
    light_position_y: float,
    light_position_z: float,
    num_source_rays: int,
    source_intensity: float = 1.0,
    source_ray_seed: int | None = None,
    source_sampling_mode: str = "point_uniform",
    n_from: float = 1.0,
    mu_s: float,
    mu_a: float,
    refractive_index: float,
    source_deposition_method: str,
    algorithm_version: str = CLEAN_OPTICAL_ALGORITHM_VERSION,
    package_version_str: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build the normalized clean optical cache-key payload.

    Camera, particle, Robin boundary condition/extrapolation, corruption, image format,
    display JPEG preview quality, and ``max_workers`` must not be included.
    """
    if not stl_sha256 and not stl_path:
        raise ValueError(
            "clean_optical_cache_key requires stl_sha256 or stl_path."
        )

    payload: dict[str, Any] = {
        "algorithm_version": _normalize_str(algorithm_version),
        "diffusion_mesh_generation_identity": _normalize_str(
            diffusion_mesh_generation_identity
        ),
        "illumination": {
            "kind": _normalize_str(illumination_kind),
            "light_position": [
                _normalize_float(light_position_x),
                _normalize_float(light_position_y),
                _normalize_float(light_position_z),
            ],
            "source_intensity": _normalize_float(source_intensity),
            "target_footprint": "derived_from_mesh_bounds",
        },
        "key_kind": "clean_optical",
        "material": {
            "mu_a": _normalize_float(mu_a),
            "mu_s": _normalize_float(mu_s),
            "refractive_index": _normalize_float(refractive_index),
        },
        "package_version": _normalize_str(
            package_version_str
            if package_version_str is not None
            else package_version
        ),
        "source_deposition_method": _normalize_str(source_deposition_method),
        "source_ray_schedule": {
            "num_source_rays": _normalize_int(num_source_rays),
            "sampling_mode": _normalize_str(source_sampling_mode),
            "seed": (
                None if source_ray_seed is None else _normalize_int(source_ray_seed)
            ),
        },
        "n_from": _normalize_float(n_from),
        "surface_mesh_processing_version": _normalize_str(
            surface_mesh_processing_version
        ),
    }

    if stl_sha256:
        payload["stl_sha256"] = _normalize_str(stl_sha256)
    else:
        payload["stl_path"] = _normalize_str(stl_path)
        payload["stl_sha256"] = None

    if git_commit is not None:
        payload["git_commit"] = _normalize_str(git_commit)

    return payload


def clean_optical_cache_key(**kwargs: Any) -> str:
    """Return the full SHA256 clean optical cache ID."""
    return hash_cache_payload(clean_optical_cache_key_payload(**kwargs))


def _particle_item_payload(
    *,
    particle_setup_id: str | None,
    particle_kind: str,
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float,
    mu_s_particle: float,
    mu_a_particle: float,
    refractive_index_particle: float,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "center": [
            _normalize_float(center_x),
            _normalize_float(center_y),
            _normalize_float(center_z),
        ],
        "kind": _normalize_str(particle_kind),
        "mu_a_particle": _normalize_float(mu_a_particle),
        "mu_s_particle": _normalize_float(mu_s_particle),
        "radius": _normalize_float(radius),
        "refractive_index_particle": _normalize_float(refractive_index_particle),
    }
    if particle_setup_id is not None:
        item["particle_setup_id"] = _normalize_str(particle_setup_id)
    return item


def _normalize_particle_items(
    particles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize an ordered particle list. Workbook order is scientific."""
    if not particles:
        raise ValueError("particle_source_cache_key requires a non-empty particles list.")
    items: list[dict[str, Any]] = []
    for index, raw in enumerate(particles):
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"particles[{index}] must be a mapping, got {type(raw)!r}"
            )
        try:
            items.append(
                _particle_item_payload(
                    particle_setup_id=(
                        None
                        if raw.get("particle_setup_id") is None
                        else str(raw["particle_setup_id"])
                    ),
                    particle_kind=str(raw["particle_kind"]),
                    center_x=float(raw["center_x"]),
                    center_y=float(raw["center_y"]),
                    center_z=float(raw["center_z"]),
                    radius=float(raw["radius"]),
                    mu_s_particle=float(raw["mu_s_particle"]),
                    mu_a_particle=float(raw["mu_a_particle"]),
                    refractive_index_particle=float(raw["refractive_index_particle"]),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"particles[{index}] missing required field {exc.args[0]!r}"
            ) from exc
    return items


def particle_source_cache_key_payload(
    *,
    clean_optical_cache_id: str,
    particles: Sequence[Mapping[str, Any]] | None = None,
    particle_kind: str | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    center_z: float | None = None,
    radius: float | None = None,
    mu_s_particle: float | None = None,
    mu_a_particle: float | None = None,
    refractive_index_particle: float | None = None,
    particle_setup_id: str | None = None,
    placement_mode: str,
    seed: int | None = None,
    source_delta_assignment: str = "attenuated_chord",
    algorithm_version: str = PARTICLE_SOURCE_ALGORITHM_VERSION,
) -> dict[str, Any]:
    """Build the normalized particle source cache-key payload.

    Prefer ``particles`` as an ordered list of particle mappings. Workbook
    order is part of the scientific key (not sorted). Legacy single-particle
    kwargs still build a one-element list.

    Camera, Robin boundary condition/extrapolation, corruption, image format, display JPEG
    preview quality, and ``max_workers`` must not be included.
    """
    if particles is None:
        missing = [
            name
            for name, value in (
                ("particle_kind", particle_kind),
                ("center_x", center_x),
                ("center_y", center_y),
                ("center_z", center_z),
                ("radius", radius),
                ("mu_s_particle", mu_s_particle),
                ("mu_a_particle", mu_a_particle),
                ("refractive_index_particle", refractive_index_particle),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "particle_source_cache_key requires particles=... or "
                f"single-particle fields; missing {missing}"
            )
        particle_items = [
            _particle_item_payload(
                particle_setup_id=particle_setup_id,
                particle_kind=str(particle_kind),
                center_x=float(center_x),
                center_y=float(center_y),
                center_z=float(center_z),
                radius=float(radius),
                mu_s_particle=float(mu_s_particle),
                mu_a_particle=float(mu_a_particle),
                refractive_index_particle=float(refractive_index_particle),
            )
        ]
    else:
        particle_items = _normalize_particle_items(particles)

    placement_mode_norm = _normalize_str(placement_mode)
    payload: dict[str, Any] = {
        "algorithm_version": _normalize_str(algorithm_version),
        "clean_optical_cache_id": _normalize_str(clean_optical_cache_id),
        "key_kind": "particle_source",
        "particles": particle_items,
        "placement_mode": placement_mode_norm,
        "source_delta_assignment": _normalize_str(source_delta_assignment),
    }

    if placement_mode_norm != "fixed":
        if seed is None:
            raise ValueError(
                "particle_source_cache_key requires seed when "
                f"placement_mode={placement_mode_norm!r}."
            )
        payload["seed"] = _normalize_int(seed)
    elif seed is not None:
        # Fixed placements may still record a seed for provenance, but it
        # must not affect the scientific key unless placement is randomized.
        pass

    return payload


def particle_source_cache_key(**kwargs: Any) -> str:
    """Return the full SHA256 particle source cache ID."""
    return hash_cache_payload(particle_source_cache_key_payload(**kwargs))


def camera_visibility_cache_key_payload(
    *,
    stl_sha256: str | None = None,
    stl_path: str | None = None,
    surface_mesh_processing_version: str = SURFACE_MESH_PROCESSING_VERSION,
    look_at_policy: str = CAMERA_LOOK_AT_POLICY,
    camera_kind: str,
    fov_deg: float,
    resolution_x: int,
    resolution_y: int,
    frame_index: int,
    angle_deg: float,
    axis_x: float,
    axis_y: float,
    axis_z: float,
    distance: float,
    elevation_deg: float,
    lateral_offset: float | None = None,
    z_offset: float | None = None,
    up_variant: str | None = None,
    algorithm_version: str = CAMERA_VISIBILITY_ALGORITHM_VERSION,
    package_version_str: str | None = None,
) -> dict[str, Any]:
    """Build the normalized camera×mesh visibility cache-key payload.

    First-surface hits depend only on STL triangle mesh file identity, camera intrinsics, and
    pose. Optical properties, particles, Robin boundary condition/extrapolation, fluence field ``Phi``, image
    format, display JPEG preview quality, and ``max_workers`` must not be included.
    """
    if not stl_sha256 and not stl_path:
        raise ValueError(
            "camera_visibility_cache_key requires stl_sha256 or stl_path."
        )

    pose: dict[str, Any] = {
        "angle_deg": _normalize_float(angle_deg),
        "axis": [
            _normalize_float(axis_x),
            _normalize_float(axis_y),
            _normalize_float(axis_z),
        ],
        "distance": _normalize_float(distance),
        "elevation_deg": _normalize_float(elevation_deg),
        "frame_index": _normalize_int(frame_index),
        "lateral_offset": (
            None if lateral_offset is None else _normalize_float(lateral_offset)
        ),
        "up_variant": None if up_variant is None else _normalize_str(up_variant),
        "z_offset": None if z_offset is None else _normalize_float(z_offset),
    }
    payload: dict[str, Any] = {
        "algorithm_version": _normalize_str(algorithm_version),
        "camera": {
            "fov_deg": _normalize_float(fov_deg),
            "kind": _normalize_str(camera_kind),
            "resolution_x": _normalize_int(resolution_x),
            "resolution_y": _normalize_int(resolution_y),
        },
        "key_kind": "camera_visibility",
        "look_at_policy": _normalize_str(look_at_policy),
        "package_version": _normalize_str(
            package_version_str
            if package_version_str is not None
            else package_version
        ),
        "pose": pose,
        "surface_mesh_processing_version": _normalize_str(
            surface_mesh_processing_version
        ),
    }
    if stl_sha256:
        payload["stl_sha256"] = _normalize_str(stl_sha256)
    else:
        payload["stl_path"] = _normalize_str(stl_path)
        payload["stl_sha256"] = None
    return payload


def camera_visibility_cache_key(**kwargs: Any) -> str:
    """Return the full SHA256 camera×mesh visibility cache ID."""
    return hash_cache_payload(camera_visibility_cache_key_payload(**kwargs))


def phi_sampling_localization_cache_key_payload(
    *,
    camera_visibility_cache_id: str,
    diffusion_mesh_content_hash: str,
    diffusion_mesh_num_nodes: int,
    diffusion_mesh_num_tets: int,
    interpolation_method: str = "tetrahedral_barycentric",
    barycentric_tolerance: float = DEFAULT_BARYCENTRIC_TOLERANCE,
    algorithm_version: str = PHI_SAMPLING_LOCALIZATION_ALGORITHM_VERSION,
    package_version_str: str | None = None,
) -> dict[str, Any]:
    """Build the fluence field ``Phi`` sampling localization cache-key payload.

    Localization depends only on the diffusion mesh and the camera hit map
    (via ``camera_visibility_cache_id``). Optical properties, particles,
    ``Phi``, Robin boundary condition settings, image format, and ``max_workers`` must not appear.
    """
    return {
        "algorithm_version": _normalize_str(algorithm_version),
        "camera_visibility_cache_id": _normalize_str(camera_visibility_cache_id),
        "diffusion_mesh": {
            "content_hash": _normalize_str(diffusion_mesh_content_hash),
            "num_nodes": _normalize_int(diffusion_mesh_num_nodes),
            "num_tets": _normalize_int(diffusion_mesh_num_tets),
        },
        "interpolation": {
            "barycentric_tolerance": _normalize_float(barycentric_tolerance),
            "method": _normalize_str(interpolation_method),
        },
        "key_kind": "phi_sampling_localization",
        "package_version": _normalize_str(
            package_version_str
            if package_version_str is not None
            else package_version
        ),
    }


def phi_sampling_localization_cache_key(**kwargs: Any) -> str:
    """Return the full SHA256 fluence field ``Phi`` sampling localization cache ID."""
    return hash_cache_payload(phi_sampling_localization_cache_key_payload(**kwargs))


def diffusion_settings_provenance(
    *,
    diffusion_setup_id: str,
    D: float,
    mu_a: float,
    robin_boundary_model: str,
    extrapolation_length: float,
    fem_order: int,
    solver_tolerance: float,
    alpha_direct: float,
    mu_a_source: str = "optical_setup",
    D_source: str = "optical_material_diffusion_coefficient",
    optical_setup_id: str | None = None,
    g: float | None = None,
    g_source: str = "diffusion_setup",
) -> dict[str, Any]:
    """Return diffusion/composition provenance (not a persistent operator cache).

    ``mu_a`` comes from the linked optical setup. ``g`` comes from the
    diffusion setup. Callers must obtain ``D`` from
    ``OpticalMaterialConfig.diffusion_coefficient`` after building that
    material from optical scattering coefficient ``mu_s``/absorption coefficient ``mu_a`` and diffusion-setup ``g``. Do not
    recompute ``D`` from a local formula here; the optics material property is
    the model authority. Neither ``D`` nor ``g`` belongs on the optical sheet;
    ``g`` must not invalidate the clean optical cache.

    ``alpha_direct`` is the hybrid-composition scale
    ``I_total = alpha_direct * I_direct + I_diffuse``.
    """
    payload: dict[str, Any] = {
        "D": _normalize_float(D),
        "D_source": _normalize_str(D_source),
        "alpha_direct": _normalize_float(alpha_direct),
        "diffusion_setup_id": _normalize_str(diffusion_setup_id),
        "extrapolation_length": _normalize_float(extrapolation_length),
        "fem_order": _normalize_int(fem_order),
        "mu_a": _normalize_float(mu_a),
        "mu_a_source": _normalize_str(mu_a_source),
        "provenance_kind": "diffusion_settings",
        "robin_boundary_model": _normalize_str(robin_boundary_model),
        "solver_tolerance": _normalize_float(solver_tolerance),
    }
    if optical_setup_id is not None:
        payload["optical_setup_id"] = _normalize_str(optical_setup_id)
    if g is not None:
        payload["g"] = _normalize_float(g)
        payload["g_source"] = _normalize_str(g_source)
    return payload
