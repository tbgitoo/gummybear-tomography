"""First-surface ray intersection and camera hit-image utilities."""

from typing import Any

import numpy as np
import trimesh


def first_visible_hits(
    mesh: trimesh.Trimesh,
    rays: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the first visible surface hit for each ray in a bundle.

    Intersects ``mesh`` with flat ``origins`` / ``directions`` arrays. Hit depth
    is the distance from the ray origin to the first surface along the ray — not
    internal path length or object thickness.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Triangle mesh to intersect.
    rays : RayBundleProtocol
        Object with ``origins`` and ``directions`` of shape ``(N, 3)`` and
        optional ``sample_shape``. Compatible with ``CameraRayBundle`` and
        ``SourceRayBundle``.

    Returns
    -------
    valid_mask : np.ndarray, shape (N,), dtype bool
        ``True`` where ray ``i`` hits the mesh.
    hit_depth : np.ndarray, shape (N,), dtype float
        Distance from origin to first hit; ``NaN`` on miss.
    hit_face : np.ndarray, shape (N,), dtype int
        Hit triangle index, or ``-1`` on miss.

    Notebook / protocol: M2A camera pass.

    See also:
        :func:`first_visible_hits_with_points` — adds world-space hit locations for diffuse sampling.
        :func:`~gummybear.optics.face_illumination.sample_face_values_to_image` — face-indexed illumination pass.
    """
    ray_origins = np.asarray(rays.origins, dtype=float)
    ray_directions = np.asarray(rays.directions, dtype=float)

    if ray_origins.ndim != 2 or ray_origins.shape[1] != 3:
        raise ValueError(f"rays.origins must have shape [N, 3], got {ray_origins.shape}")

    if ray_directions.ndim != 2 or ray_directions.shape[1] != 3:
        raise ValueError(
            f"rays.directions must have shape [N, 3], got {ray_directions.shape}"
        )

    if ray_origins.shape != ray_directions.shape:
        raise ValueError(
            "rays.origins and rays.directions must have the same shape, "
            f"got {ray_origins.shape} and {ray_directions.shape}"
        )

    sample_shape = getattr(rays, "sample_shape", None)
    if sample_shape is not None:
        expected_n = int(np.prod(sample_shape))
        if ray_origins.shape[0] != expected_n:
            raise ValueError(
                "Number of rays must match sample_shape: "
                f"got N={ray_origins.shape[0]}, sample_shape={sample_shape} "
                f"(expected N={expected_n})"
            )

    n_rays = ray_origins.shape[0]

    # Normalize directions so that depth is a true distance in mesh units.
    norms = np.linalg.norm(ray_directions, axis=1)

    if np.any(norms == 0):
        raise ValueError("rays.directions contains at least one zero-length vector.")

    ray_directions = ray_directions / norms[:, None]

    valid_mask = np.zeros(n_rays, dtype=bool)
    hit_depth = np.full(n_rays, np.nan, dtype=float)
    hit_face = np.full(n_rays, -1, dtype=int)

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False,
    )

    if len(index_ray) == 0:
        return valid_mask, hit_depth, hit_face

    # Since directions are normalized, the dot product gives distance along ray.
    vectors_to_hit = locations - ray_origins[index_ray]
    depths = np.einsum("ij,ij->i", vectors_to_hit, ray_directions[index_ray])

    # Keep only positive-depth hits. Usually trimesh already does this,
    # but this protects against numerical weirdness.
    positive = depths > 0

    index_ray = index_ray[positive]
    index_tri = index_tri[positive]
    depths = depths[positive]

    valid_mask[index_ray] = True
    hit_depth[index_ray] = depths
    hit_face[index_ray] = index_tri

    return valid_mask, hit_depth, hit_face


def first_visible_hits_with_points(mesh, rays):
    """Return first hits plus world-space hit locations.

    Wraps :func:`first_visible_hits` and reconstructs 3D hit points as
    ``origins + depth * directions``. Missed rays receive ``NaN`` coordinates.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Triangle mesh to intersect.
    rays : RayBundleProtocol
        Ray bundle with ``origins`` and ``directions`` of shape ``(N, 3)``.

    Returns
    -------
    valid_mask : np.ndarray, shape (N,), dtype bool
        Hit validity mask.
    hit_depth : np.ndarray, shape (N,), dtype float
        First-surface depth along each ray.
    hit_face : np.ndarray, shape (N,), dtype int
        Hit triangle index, or ``-1`` on miss.
    points : np.ndarray, shape (N, 3), dtype float
        World-space hit locations; ``NaN`` rows for misses.

    Notebook / protocol: M2A camera pass.

    See also:
        :func:`~gummybear.optics.diffuse_sampling.sample_phi_at_hit_points` — fluence ``Phi`` at hit locations.
        :func:`~gummybear.optics.diffuse_sampling.sample_diffuse_image` — ``I_diffuse`` camera channel.
    """
    valid, depth, faces = first_visible_hits(mesh, rays)
    points = rays.origins + depth[:, None] * rays.directions
    points[~valid] = np.nan
    return valid, depth, faces, points


def hits_to_image(
    mesh: trimesh.Trimesh,
    valid_mask: np.ndarray,
    hit_depth: np.ndarray,
    hit_faces: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape flat first-hit results into square image arrays.

    Assumes rays were generated on a ``resolution × resolution`` grid in the
    same row-major order as :func:`make_camera_rays`.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh used for intersection (face normals are looked up here).
    valid_mask : np.ndarray, shape (N,)
        Flat boolean hit mask with ``N = resolution ** 2``.
    hit_depth : np.ndarray, shape (N,)
        Flat first-surface depths; ``NaN`` for misses.
    hit_faces : np.ndarray, shape (N,)
        Flat hit triangle indices; ``-1`` for misses.
    resolution : int
        Square image side length ``H == W``.

    Returns
    -------
    mask_image : np.ndarray, shape (H, W), dtype bool
        Per-pixel hit validity.
    depth_image : np.ndarray, shape (H, W), dtype float
        First-surface depth image (not thickness).
    normal_image : np.ndarray, shape (H, W, 3), dtype float
        Face normal at each hit; ``NaN`` for misses.

    Notebook / protocol: M2A camera pass.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    hit_depth = np.asarray(hit_depth, dtype=float)
    hit_faces = np.asarray(hit_faces, dtype=int)

    expected_n = resolution * resolution

    if valid_mask.shape != (expected_n,):
        raise ValueError(
            f"valid_mask must have shape ({expected_n},), got {valid_mask.shape}"
        )

    if hit_depth.shape != (expected_n,):
        raise ValueError(
            f"hit_depth must have shape ({expected_n},), got {hit_depth.shape}"
        )

    if hit_faces.shape != (expected_n,):
        raise ValueError(
            f"hit_faces must have shape ({expected_n},), got {hit_faces.shape}"
        )

    mask_image = valid_mask.reshape(resolution, resolution)
    depth_image = hit_depth.reshape(resolution, resolution)

    normal_flat = np.full((expected_n, 3), np.nan, dtype=float)

    hit_pixels = valid_mask & (hit_faces >= 0)

    normal_flat[hit_pixels] = mesh.face_normals[hit_faces[hit_pixels]]

    normal_image = normal_flat.reshape(resolution, resolution, 3)

    return mask_image, depth_image, normal_image


def simple_camera_intensity(
    mask: np.ndarray,
    beam_direction: tuple[float, float, float],
    view_direction: tuple[float, float, float],
    mode: str = "bdotv",
) -> np.ndarray:
    """Build a diagnostic camera-intensity image from a silhouette mask.

    This is not a physical appearance model. It scales the binary mask by a
    constant or by a global beam–view alignment term for quick visualization.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W)
        Boolean or numeric silhouette (non-zero treated as visible).
    beam_direction : tuple[float, float, float]
        Illumination direction **b** (normalized internally).
    view_direction : tuple[float, float, float]
        Camera view direction **v** (normalized internally).
    mode : {"constant", "bdotv"}, optional
        ``"constant"`` returns the mask unchanged; ``"bdotv"`` multiplies by
        ``(b·v + 1) / 2`` mapped to ``[0, 1]``.

    Returns
    -------
    np.ndarray, shape (H, W), dtype float
        Intensity image on ``[0, 1]`` inside the silhouette.

    Notebook / protocol: M2A camera pass.
    """
    mask = mask.astype(float)

    if mode == "constant":
        return mask

    if mode == "bdotv":

        b = np.asarray(beam_direction, dtype=float)
        v = np.asarray(view_direction, dtype=float)

        b /= np.linalg.norm(b)
        v /= np.linalg.norm(v)

        # map [-1,1] -> [0,1]
        intensity = (np.dot(b, v) + 1.0) / 2.0

        return intensity * mask

    raise ValueError(
        f"Unsupported mode '{mode}'. "
        "Expected 'constant' or 'bdotv'."
    )


def normalize_direction_sums(direction_sum):
    """Normalize per-ray summed direction vectors, leaving zeros unchanged.

    Rows with zero norm remain zero rather than producing ``NaN`` values.

    Parameters
    ----------
    direction_sum : np.ndarray, shape (N, 3)
        Accumulated direction vectors to normalize row-wise.

    Returns
    -------
    np.ndarray, shape (N, 3), dtype float
        Row-normalized directions; zero-norm rows stay zero.
    """
    out = np.zeros_like(direction_sum, dtype=float)
    norms = np.linalg.norm(direction_sum, axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            direction_sum,
            norms,
            out=out,
            where=norms > 0,
        )

    return out
