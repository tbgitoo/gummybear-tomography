"""Generate geometry-only source ray bundles from light configurations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import normalize_vector
from ..rays.source import SourceRayBundle
from .light_source import DirectionalLightConfig, LightConfig, PointLightConfig


@dataclass(frozen=True)
class SourceSamplingParams:
    """Sampling policy when converting :class:`LightConfig` to :class:`SourceRayBundle`.

    Parameters
    ----------
    mode:
        Reserved dispatch hint; directional lights use a plane grid,
        point lights use ``n_rays`` sphere/bbox samples.
    direction_n_samples:
        Grid side length for directional source plane sampling.
    n_rays:
        Number of rays for point-source sampling.
    seed:
        RNG seed for reproducible point-source targets.
    """

    mode: str = "directional_grid"  # "directional_grid" | "point_uniform" | "point_sphere"
    direction_n_samples: int = 64
    n_rays: int = 4096
    seed: int | None = None


def _plane_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal (right, up) spanning the plane perpendicular to direction."""
    d = normalize_vector(direction)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(d, up_hint))) > 0.9:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=float)
    right = np.cross(up_hint, d)
    right = right / np.linalg.norm(right)
    up = np.cross(d, right)
    up = up / np.linalg.norm(up)
    return right, up


def generate_directional_source_rays(
    light: DirectionalLightConfig,
    bbox: np.ndarray,
    grid_dim: int = 64,
    footprint_margin: float = 0.10,
    upstream_offset_fraction: float = 0.05,
) -> SourceRayBundle:
    """Build parallel source rays from an upstream plane covering the mesh bbox.

    Constructs an orthonormal plane perpendicular to ``light.propagation_direction``,
    projects all eight bbox corners onto that plane, adds ``footprint_margin`` padding,
    and places the plane slightly upstream of the bbox along the propagation direction.
    Ray weights split ``light.intensity`` uniformly across the ``grid_dim × grid_dim`` grid.

    No material interaction or mesh intersection is performed.

    Parameters
    ----------
    light:
        Directional light with propagation direction and intensity.
    bbox:
        Mesh axis-aligned bounds, shape ``[2, 3]`` (min/max corners).
    grid_dim:
        Samples per axis on the source plane (total ``grid_dim**2`` rays).
    footprint_margin:
        Fractional padding on projected bbox extent on the source plane.
    upstream_offset_fraction:
        Upstream offset as a fraction of bbox diagonal along propagation.

    Returns
    -------
    SourceRayBundle
        Origins on the source plane, parallel directions, uniform weight split.
        ``sample_shape`` is ``(grid_dim, grid_dim)``.

    Raises
    ------
    ValueError
        Invalid bbox, zero extent, or negative margin/offset parameters.
    """
    bbox = np.asarray(bbox, dtype=float)

    if bbox.shape != (2, 3):
        raise ValueError(f"bbox must have shape [2, 3], got {bbox.shape}")

    if grid_dim < 1:
        raise ValueError(f"grid_dim must be >= 1, got {grid_dim}")

    if footprint_margin < 0:
        raise ValueError(
            f"footprint_margin must be >= 0, got {footprint_margin}"
        )

    if upstream_offset_fraction < 0:
        raise ValueError(
            "upstream_offset_fraction must be >= 0, "
            f"got {upstream_offset_fraction}"
        )

    # Physical propagation direction.
    # A source ray travels as:
    #     p(t) = origin + t * d
    d = normalize_vector(light.propagation_direction)

    bbox_min = bbox[0]
    bbox_max = bbox[1]
    extent = bbox_max - bbox_min

    diagonal = float(np.linalg.norm(extent))
    if diagonal == 0.0:
        raise ValueError("mesh bbox has zero extent")

    # Orthonormal basis of the source plane.
    # right and up are perpendicular to d.
    right, up = _plane_basis(d)

    # All 8 bbox corners.
    corners = np.array(
        [
            [x, y, z]
            for x in (bbox_min[0], bbox_max[0])
            for y in (bbox_min[1], bbox_max[1])
            for z in (bbox_min[2], bbox_max[2])
        ],
        dtype=float,
    )

    corner_r = corners @ right
    corner_u = corners @ up
    corner_s = corners @ d

    r_min = float(corner_r.min())
    r_max = float(corner_r.max())
    u_min = float(corner_u.min())
    u_max = float(corner_u.max())
    s_min = float(corner_s.min())
    s_max = float(corner_s.max())

    upstream_offset = upstream_offset_fraction * diagonal
    s_plane = s_min - upstream_offset

    r_extent = r_max - r_min
    u_extent = u_max - u_min

    r_pad = footprint_margin * r_extent
    u_pad = footprint_margin * u_extent

    r_vals = np.linspace(r_min - r_pad, r_max + r_pad, grid_dim)
    u_vals = np.linspace(u_min - u_pad, u_max + u_pad, grid_dim)

    R, U = np.meshgrid(r_vals, u_vals, indexing="xy")

    origins = (
        R.ravel()[:, None] * right
        + U.ravel()[:, None] * up
        + s_plane * d
    )

    n = origins.shape[0]

    directions = np.tile(d, (n, 1))

    weights = np.full(
        n,
        float(light.intensity) / float(n),
        dtype=float,
    )

    return SourceRayBundle(
        origins=origins,
        directions=directions,
        weights=weights,
        sample_shape=(grid_dim, grid_dim),
        metadata={
            "light_type": "directional",
            "propagation_direction": tuple(map(float, d)),
            "grid_dim": int(grid_dim),
            "footprint_margin": float(footprint_margin),
            "upstream_offset_fraction": float(upstream_offset_fraction),
            "upstream_offset": float(upstream_offset),
            "s_plane": float(s_plane),
            "bbox_s_min": float(s_min),
            "bbox_s_max": float(s_max),
            "r_min": float(r_min),
            "r_max": float(r_max),
            "u_min": float(u_min),
            "u_max": float(u_max),
            "bbox": bbox.tolist(),
        },
    )


def generate_point_source_rays(
    light: PointLightConfig,
    bbox: np.ndarray,
    n_rays: int = 1024,
    seed: int | None = None,
) -> SourceRayBundle:
    """Build rays from a point source toward random targets inside the mesh bbox.

    Each ray originates at ``light.position``, points toward a uniform random
    target in ``bbox``, and receives a weight from ``light.falloff`` and
    ``light.intensity``. Degenerate zero-length directions are resampled once.

    Parameters
    ----------
    light:
        Point light with position, intensity, and falloff mode.
    bbox:
        Mesh axis-aligned bounds, shape ``[2, 3]``.
    n_rays:
        Number of rays to emit.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    SourceRayBundle
        ``sample_shape`` is ``None`` (unstructured ray list).

    Raises
    ------
    ValueError
        Invalid bbox, ``n_rays < 1``, or inability to form non-zero directions.
    """
    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape != (2, 3):
        raise ValueError(f"bbox must have shape [2, 3], got {bbox.shape}")
    if n_rays < 1:
        raise ValueError(f"n_rays must be >= 1, got {n_rays}")

    source = np.asarray(light.position, dtype=float).reshape(3)
    rng = np.random.default_rng(seed)
    targets = rng.uniform(low=bbox[0], high=bbox[1], size=(n_rays, 3))

    directions = targets - source
    distances = np.linalg.norm(directions, axis=1)
    valid = distances > 0.0
    if not np.any(valid):
        raise ValueError("All point-source directions have zero length")

    if not np.all(valid):
        n_bad = int(np.sum(~valid))
        replacements = rng.uniform(low=bbox[0], high=bbox[1], size=(n_bad, 3))
        directions[~valid] = replacements - source
        distances = np.linalg.norm(directions, axis=1)
        if np.any(distances == 0.0):
            raise ValueError("Unable to generate non-zero point-source directions")

    directions = directions / distances[:, None]
    origins = np.tile(source, (n_rays, 1))

    r = np.maximum(distances, float(light.r_min))
    if light.falloff == "inverse_square":
        weights = float(light.intensity) / (r * r)
    elif light.falloff == "none" or light.falloff is None:
        weights = np.full(n_rays, float(light.intensity) / float(n_rays), dtype=float)
    else:
        weights = float(light.intensity) / (r * r)

    return SourceRayBundle(
        origins=origins,
        directions=directions,
        weights=weights,
        sample_shape=None,
        metadata={
            "light_type": "point",
            "position": tuple(map(float, source)),
            "n_rays": n_rays,
            "falloff": light.falloff,
        },
    )


def make_source_ray_bundle(
    light: LightConfig,
    mesh_bbox: np.ndarray,
    sampling: SourceSamplingParams | None = None,
) -> SourceRayBundle:
    """Convert a light configuration into a geometry-only :class:`SourceRayBundle`.

    Dispatches directional lights to :func:`generate_directional_source_rays`
    and point lights to :func:`generate_point_source_rays`. No mesh intersection,
    refraction, or material attenuation is applied.

    Parameters
    ----------
    light:
        :class:`DirectionalLightConfig` or :class:`PointLightConfig`.
    mesh_bbox:
        Axis-aligned mesh bounds, shape ``[2, 3]``.
    sampling:
        Grid size / ray count / seed; defaults to :class:`SourceSamplingParams()`.

    Returns
    -------
    SourceRayBundle

    Raises
    ------
    TypeError
        Unsupported ``light`` type.
    """
    if sampling is None:
        sampling = SourceSamplingParams()

    if isinstance(light, DirectionalLightConfig):
        if sampling.mode not in ("directional_grid", "auto", "directional"):
            pass
        return generate_directional_source_rays(
            light=light,
            bbox=mesh_bbox,
            grid_dim=int(sampling.direction_n_samples),
        )

    if isinstance(light, PointLightConfig):
        return generate_point_source_rays(
            light=light,
            bbox=mesh_bbox,
            n_rays=int(sampling.n_rays),
            seed=sampling.seed,
        )

    raise TypeError(f"Unsupported light config type: {type(light)}")
