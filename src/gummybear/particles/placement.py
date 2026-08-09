"""Offline particle centre placement helpers.

Runtime M6/M8 generation stays ``placement_mode=fixed``. These helpers author
fixed centres (for Excel / ``attach_particle_group``) by sampling inside a
watertight mesh volume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import trimesh

from gummybear.geometry import load_stl

MeshLike = Union[str, Path, trimesh.Trimesh]


def _as_trimesh(mesh: MeshLike) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    return load_stl(mesh)


def sample_random_centers_in_mesh(
    mesh: MeshLike,
    n: int,
    *,
    seed: int | None = None,
    min_center_separation: float | None = None,
    radius: float | None = None,
    max_attempts: int | None = None,
) -> np.ndarray:
    """Sample ``n`` random centres uniformly inside a mesh volume.

    Parameters
    ----------
    mesh
        STL path (``str`` / ``Path``) or an already-loaded ``trimesh.Trimesh``.
        The mesh must be watertight so ``mesh.contains`` is well-defined.
    n
        Number of centres to return.
    seed
        Optional RNG seed for reproducibility.
    min_center_separation
        Optional minimum Euclidean distance between any two accepted centres.
        When omitted and ``radius`` is set, defaults to ``2 * radius`` so equal
        spheres of that radius do not geometrically overlap.
    radius
        Convenience radius for equal-sphere non-overlap (see above). Does not
        enforce that the full sphere lies inside the mesh; only the centre is
        required to be inside (boundary-straddling spheres remain allowed).
    max_attempts
        Maximum rejection-sampling draws. Defaults to
        ``max(10_000, 200 * n * (1 + sep_scale))`` where ``sep_scale`` grows
        with requested separation relative to mesh size.

    Returns
    -------
    centers : ndarray, shape ``(n, 3)``
        Float64 centres inside the mesh.

    Raises
    ------
    ValueError
        If ``n`` is invalid, the mesh is not watertight, or sampling fails to
        find ``n`` valid centres within ``max_attempts``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.zeros((0, 3), dtype=float)

    tri = _as_trimesh(mesh)
    if not bool(tri.is_watertight):
        raise ValueError(
            "sample_random_centers_in_mesh requires a watertight mesh "
            "(mesh.contains is otherwise undefined)."
        )

    if radius is not None:
        radius = float(radius)
        if radius < 0.0:
            raise ValueError("radius must be non-negative")

    if min_center_separation is None:
        sep = (2.0 * radius) if radius is not None else 0.0
    else:
        sep = float(min_center_separation)
        if sep < 0.0:
            raise ValueError("min_center_separation must be non-negative")

    extents = np.asarray(tri.extents, dtype=float)
    characteristic = float(np.linalg.norm(extents))
    sep_scale = (sep / characteristic) if characteristic > 0.0 and sep > 0.0 else 0.0
    if max_attempts is None:
        max_attempts = max(10_000, int(200 * n * (1.0 + 20.0 * sep_scale)))
    else:
        max_attempts = int(max_attempts)
        if max_attempts < n:
            raise ValueError("max_attempts must be at least n")

    rng = np.random.default_rng(seed)
    bounds0 = np.asarray(tri.bounds[0], dtype=float)
    accepted: list[np.ndarray] = []

    # Draw in batches to amortize contains() cost.
    remaining_attempts = max_attempts
    while len(accepted) < n and remaining_attempts > 0:
        batch = min(max(64, 4 * (n - len(accepted))), remaining_attempts)
        remaining_attempts -= batch
        candidates = bounds0 + rng.random((batch, 3)) * extents
        inside = np.asarray(tri.contains(candidates), dtype=bool)
        for point in candidates[inside]:
            if len(accepted) >= n:
                break
            if sep > 0.0 and accepted:
                dists = np.linalg.norm(
                    np.asarray(accepted, dtype=float) - point[None, :],
                    axis=1,
                )
                if float(np.min(dists)) < sep:
                    continue
            accepted.append(np.asarray(point, dtype=float).reshape(3))

    if len(accepted) < n:
        raise ValueError(
            f"Failed to sample {n} centres inside the mesh within "
            f"{max_attempts} attempts "
            f"(accepted={len(accepted)}, min_center_separation={sep}). "
            "Reduce n/separation or increase max_attempts."
        )

    return np.asarray(accepted, dtype=float).reshape(n, 3)
