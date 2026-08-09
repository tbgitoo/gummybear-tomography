"""Upstream material thickness and Beer–Lambert transmittance on mesh faces."""

from typing import Union

import numpy as np
import trimesh

from .light_source import (
    DirectionalLightConfig,
    PointLightConfig,
    illumination_directions_at_faces,
)

from ..geometry import face_centroids


def _unique_sorted_distances(t_values: np.ndarray, tol: float) -> np.ndarray:
    """
    Sort ray-intersection distances and merge near-duplicates.

    Near-duplicate intersections happen easily when a ray hits a triangle edge
    or vertex and trimesh reports both neighboring triangles.
    """
    if len(t_values) == 0:
        return t_values

    t_sorted = np.sort(t_values)
    unique = [t_sorted[0]]

    for t in t_sorted[1:]:
        if abs(t - unique[-1]) > tol:
            unique.append(t)

    return np.asarray(unique, dtype=float)


def _accumulate_ray_material_length(
    t_values: np.ndarray,
    starts_inside: bool,
) -> float:
    """
    Accumulate material intervals along one half-ray.

    If ray starts outside:
        material intervals are [t0, t1], [t2, t3], ...

    If ray starts inside:
        material intervals are [0, t0], [t1, t2], [t3, t4], ...

    The ray is a half-ray starting at t=0 and extending toward +direction.
    """
    if len(t_values) == 0:
        return 0.0

    L = 0.0

    if starts_inside:
        # First segment from origin to first intersection is inside material.
        L += max(t_values[0], 0.0)

        # Then alternate outside/inside.
        start = 1
    else:
        start = 0

    # Pair remaining intersections.
    for i in range(start, len(t_values) - 1, 2):
        L += max(t_values[i + 1] - t_values[i], 0.0)

    return float(L)


def compute_face_upstream_thickness(
    mesh: trimesh.Trimesh,
    light: Union[DirectionalLightConfig, PointLightConfig],
    eps: float = 1e-4,
) -> np.ndarray:
    """Estimate source-side material thickness along illumination at each face.

    For face ``f`` with centroid ``P_f`` and illumination direction ``b_f``
    (source toward object), cast an upstream half-ray from
    ``P_f - eps * b_f`` in direction ``-b_f`` and accumulate in-object chord
    lengths through alternating enter/exit intersections.

    Uses ``mesh.contains`` for the ray origin when available; otherwise falls
    back to a face-normal heuristic. Edge/vertex duplicate hits are merged
    with a tolerance scaled to mesh bbox diagonal.

    Parameters
    ----------
    mesh:
        Watertight (or near-watertight) surface mesh in mesh-length units.
    light:
        Directional or point light supplying per-face ``b_f``.
    eps:
        Small offset from the face centroid along ``-b_f`` to avoid self-hits
        (mesh units).

    Returns
    -------
    L_proxy : np.ndarray, shape ``[n_faces]``
        Upstream material thickness per face (mesh units).

    Notebook / protocol:
        M2B face transmittance proxy input ``L_proxy[f]``.
    """
    centroids = face_centroids(mesh)
    b_f = illumination_directions_at_faces(mesh, light)

    n_faces = len(mesh.faces)

    if centroids.shape != (n_faces, 3):
        raise ValueError(
            f"Expected centroids shape {(n_faces, 3)}, got {centroids.shape}"
        )

    if b_f.shape != (n_faces, 3):
        raise ValueError(
            f"Expected b_f shape {(n_faces, 3)}, got {b_f.shape}"
        )

    # Upstream probe rays.
    origins = centroids - eps * b_f
    directions = -b_f

    # Tolerance scaled to mesh size.
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    t_tol = max(eps * 10.0, bbox_diag * 1e-9)

    # Decide whether each ray origin starts inside the solid.
    #
    # Preferred: mesh.contains().
    # Fallback: use face normal heuristic.
    # For a surface centroid shifted by -eps*b_f:
    #   starts_inside is approximately True when dot(b_f, outward_normal) > 0
    try:
        starts_inside = mesh.contains(origins)
    except Exception:
        starts_inside = np.einsum("ij,ij->i", b_f, mesh.face_normals) > 0.0

    # Intersect all upstream rays with the mesh.
    locations, index_ray, _index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True,
    )

    L_proxy = np.zeros(n_faces, dtype=float)

    if len(locations) == 0:
        return L_proxy

    # Convert hit locations to distances t along each ray.
    # directions are unit vectors, so dot(location - origin, direction) = distance.
    hit_origins = origins[index_ray]
    hit_dirs = directions[index_ray]
    t_all = np.einsum("ij,ij->i", locations - hit_origins, hit_dirs)

    # Keep only forward hits, away from the epsilon-near self-hit region.
    valid = t_all > t_tol
    index_ray = index_ray[valid]
    t_all = t_all[valid]

    # Accumulate per ray.
    for ray_i in np.unique(index_ray):
        t_values = t_all[index_ray == ray_i]
        t_values = _unique_sorted_distances(t_values, tol=t_tol)

        L_proxy[ray_i] = _accumulate_ray_material_length(
            t_values=t_values,
            starts_inside=bool(starts_inside[ray_i]),
        )

    # Numerical guard.
    L_proxy = np.maximum(L_proxy, 0.0)

    return L_proxy


def thickness_to_transmittance(
    thickness: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Convert material thickness to Beer–Lambert face transmittance.

    Computes ``T = exp(-mu * max(thickness, 0))`` element-wise.

    Parameters
    ----------
    thickness:
        Material path lengths, typically ``L_proxy[f]`` from
        :func:`compute_face_upstream_thickness` (mesh units).
    mu:
        Linear attenuation coefficient (1 / mesh units).

    Returns
    -------
    T_face : np.ndarray
        Transmittance values in ``(0, 1]``, same shape as ``thickness``.

    Raises
    ------
    ValueError
        If ``mu < 0``.

    Notebook / protocol:
        M2B ``T_face[f]`` sampled into camera intensity via ``hit_faces``.
    """
    thickness = np.asarray(thickness, dtype=float)

    if mu < 0:
        raise ValueError(f"mu must be non-negative, got {mu}")

    T = np.exp(-mu * np.maximum(thickness, 0.0))

    return T
