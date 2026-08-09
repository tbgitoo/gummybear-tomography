"""Snell refraction helpers for entry and exit ray transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from ..rays.source import SourceRayBundle
from ..rays.visibility import first_visible_hits_with_points


def refract_direction(
    direction: np.ndarray,
    normal: np.ndarray,
    n_from: float,
    n_to: float,
) -> tuple[np.ndarray, bool]:
    """Refract a ray direction across a dielectric interface using Snell's law.

    Orients the face normal toward the incident medium (``cos_i = -dot(d, n) > 0``)
    and returns the unit transmitted direction in the physical propagation
    direction. Returns zeros and ``ok=False`` for total internal reflection,
    degenerate normals, or zero refractive index.

    Parameters
    ----------
    direction:
        Incident ray direction, unit or near-unit, shape ``(3,)``.
    normal:
        Face unit normal (sign corrected internally).
    n_from:
        Refractive index of the incident medium (dimensionless).
    n_to:
        Refractive index of the transmitted medium.

    Returns
    -------
    transmitted : np.ndarray, shape ``(3,)``
        Unit transmitted direction, or zeros on failure.
    ok : bool
        ``True`` when refraction succeeded.
    """
    d = np.asarray(direction, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)

    d_norm = np.linalg.norm(d)
    n_norm = np.linalg.norm(n)
    if d_norm == 0.0 or n_norm == 0.0:
        return np.zeros(3, dtype=float), False

    d = d / d_norm
    n = n / n_norm

    # Orient normal toward the incident medium: cosi = -dot(d, n) > 0.
    cos_i = -float(np.dot(d, n))
    if cos_i < 0.0:
        n = -n
        cos_i = -cos_i

    if n_to == 0.0:
        return np.zeros(3, dtype=float), False

    eta = float(n_from) / float(n_to)
    sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return np.zeros(3, dtype=float), False

    cos_t = float(np.sqrt(1.0 - sin2_t))
    transmitted = eta * d + (eta * cos_i - cos_t) * n
    t_norm = np.linalg.norm(transmitted)
    if t_norm == 0.0 or not np.isfinite(t_norm):
        return np.zeros(3, dtype=float), False

    return transmitted / t_norm, True


def refract_directions(
    directions: np.ndarray,
    normals: np.ndarray,
    n_from: float,
    n_to: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`refract_direction` row-wise to batched rays and normals.

    Parameters
    ----------
    directions:
        Incident directions, shape ``[N, 3]``.
    normals:
        Face normals, same shape as ``directions``.
    n_from, n_to:
        Refractive indices of incident and transmitted media.

    Returns
    -------
    transmitted : np.ndarray, shape ``[N, 3]``
    ok : np.ndarray, shape ``[N]``, bool
    """
    directions = np.asarray(directions, dtype=float)
    normals = np.asarray(normals, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"directions must be [N, 3], got {directions.shape}")
    if normals.shape != directions.shape:
        raise ValueError(
            f"normals must match directions shape {directions.shape}, got {normals.shape}"
        )

    n = directions.shape[0]
    transmitted = np.zeros((n, 3), dtype=float)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        t, valid = refract_direction(directions[i], normals[i], n_from, n_to)
        transmitted[i] = t
        ok[i] = valid
    return transmitted, ok


@dataclass(frozen=True)
class RefractedRayBundleResult:
    """Compact bundle of rays successfully refracted at first surface hits.

    ``rays`` contains only refracted rays with origins offset slightly along
    the transmitted direction to avoid self-intersection. ``parent_indices``
    maps each output ray back to the input bundle index; full-length masks and
    hit diagnostics are retained for bookkeeping.

    Attributes
    ----------
    rays:
        Compact :class:`~gummybear.rays.source.SourceRayBundle` of successes.
    parent_indices:
        Input-ray indices for each output ray, shape ``[n_refracted]``.
    valid_mask:
        Per-input-ray success mask, shape ``[n_input]``.
    hit_faces, hit_points:
        First-surface hit data for all input rays.
    refracted_directions:
        Transmitted directions for all inputs (zeros where refraction failed).
    eps:
        Origin offset along transmitted direction (mesh units).
    metadata:
        Counts and refractive-index bookkeeping.
    """

    rays: SourceRayBundle
    parent_indices: np.ndarray
    valid_mask: np.ndarray
    hit_faces: np.ndarray
    hit_points: np.ndarray
    refracted_directions: np.ndarray
    eps: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_input(self) -> int:
        """Number of rays in the input bundle."""
        return len(self.valid_mask)

    @property
    def n_refracted(self) -> int:
        """Number of successfully refracted rays in ``rays``."""
        return self.rays.n_rays


def refract_ray_bundle(
    mesh: trimesh.Trimesh,
    rays: Any,
    *,
    n_from: float,
    n_to: float,
    eps: float | None = None,
) -> RefractedRayBundleResult:
    """Refract an entire ray bundle at first visible surface hits.

    Pipeline:

    1. ``first_visible_hits_with_points(mesh, rays)``
    2. Snell transmission at hit face normals via :func:`refract_direction`
    3. New origins at ``hit_point + eps * transmitted_direction``
    4. Compact output :class:`~gummybear.rays.source.SourceRayBundle`

    Typical index pairs: air→material ``n_from=1.0``, ``n_to=material.n_refractive``;
    material→air on exit with indices swapped.

    Parameters
    ----------
    mesh:
        Triangle mesh providing face normals and hit geometry.
    rays:
        Object with ``origins`` / ``directions`` (and optional ``weights``)
        accepted by :func:`~gummybear.rays.visibility.first_visible_hits`.
    n_from, n_to:
        Refractive indices of current and entered media.
    eps:
        Origin offset along transmitted direction. Defaults to
        ``1e-6 * mesh_bbox_diagonal``.

    Returns
    -------
    RefractedRayBundleResult
        Compact refracted bundle plus full-length hit and validity arrays.

    Notebook / protocol:
        M3A entry/exit refraction before internal transport or M4B deposition.
    """
    origins = np.asarray(rays.origins, dtype=float)
    directions = np.asarray(rays.directions, dtype=float)
    if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("rays.origins/directions must share shape [N, 3]")

    n_rays = len(origins)
    weights = getattr(rays, "weights", None)
    if weights is None:
        weights = np.ones(n_rays, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_rays,):
            raise ValueError(f"rays.weights must be [{n_rays}], got {weights.shape}")

    if eps is None:
        diag = float(np.linalg.norm(np.asarray(mesh.bounds[1] - mesh.bounds[0])))
        eps = 1e-6 * max(diag, 1e-12)
    else:
        eps = float(eps)

    hit_valid, _hit_depth, hit_faces, hit_points = first_visible_hits_with_points(
        mesh, rays
    )

    refracted_directions = np.zeros_like(directions)
    refract_ok = np.zeros(n_rays, dtype=bool)

    candidates = np.flatnonzero(hit_valid & (hit_faces >= 0))
    if candidates.size > 0:
        normals = np.asarray(mesh.face_normals[hit_faces[candidates]], dtype=float)
        transmitted, ok = refract_directions(
            directions[candidates],
            normals,
            n_from=n_from,
            n_to=n_to,
        )
        refracted_directions[candidates] = transmitted
        refract_ok[candidates] = ok

    valid_mask = hit_valid & (hit_faces >= 0) & refract_ok
    parent_indices = np.flatnonzero(valid_mask)

    if parent_indices.size == 0:
        out_rays = SourceRayBundle(
            origins=np.zeros((0, 3), dtype=float),
            directions=np.zeros((0, 3), dtype=float),
            weights=np.zeros(0, dtype=float),
            sample_shape=None,
            metadata={"refracted_from": "empty"},
        )
    else:
        out_dirs = refracted_directions[parent_indices]
        out_origins = hit_points[parent_indices] + eps * out_dirs
        out_rays = SourceRayBundle(
            origins=out_origins,
            directions=out_dirs,
            weights=weights[parent_indices],
            sample_shape=None,
            metadata={
                "n_from": float(n_from),
                "n_to": float(n_to),
                "eps": float(eps),
            },
        )

    return RefractedRayBundleResult(
        rays=out_rays,
        parent_indices=parent_indices,
        valid_mask=valid_mask,
        hit_faces=np.asarray(hit_faces, dtype=int),
        hit_points=np.asarray(hit_points, dtype=float),
        refracted_directions=refracted_directions,
        eps=float(eps),
        metadata={
            "n_from": float(n_from),
            "n_to": float(n_to),
            "n_input": int(n_rays),
            "n_hits": int(np.count_nonzero(hit_valid & (hit_faces >= 0))),
            "n_refracted": int(parent_indices.size),
        },
    )
