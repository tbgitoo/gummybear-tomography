"""Ray-to-volume scattering source deposition on a coarse diffusion mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import trimesh

from ..rays.visibility import first_visible_hits_with_points
from .diffusion_mesh import DiffusionMesh
from .material import OpticalMaterialConfig


@dataclass(frozen=True)
class SourceDepositionResult:
    """Volumetric scattering source density and energy bookkeeping from ray segments.

    ``S_clean[e]`` is element-integrated scattered energy divided by tet volume
    (volumetric source density ``S(x)`` for the diffusion solve). Only scattering contributes to
    ``S``; absorption is tracked as irreversible loss.

    Attributes
    ----------
    S_clean:
        Volumetric source density ``S(x)`` per tet, shape ``[n_tets]``.
    E_scat_elem:
        Element-integrated scattered energy, shape ``[n_tets]``.
    total_ballistic_input:
        Sum of input segment intensities.
    total_scattered, total_absorbed:
        Integrated scattered and absorbed energy along deposited segments.
    remaining_direct_energy:
        Ballistic energy not deposited (misses mesh or exits without scatter).
    mu_s, mu_a:
        Scattering coefficient ``mu_s`` and absorption coefficient ``mu_a`` used during deposition.
    """

    S_clean: np.ndarray  # [n_tets] volumetric source density
    E_scat_elem: np.ndarray  # [n_tets] element-integrated scattered energy
    total_ballistic_input: float
    total_scattered: float
    total_absorbed: float
    remaining_direct_energy: float
    mu_s: float
    mu_a: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "S_clean", np.asarray(self.S_clean, dtype=float))
        object.__setattr__(self, "E_scat_elem", np.asarray(self.E_scat_elem, dtype=float))


@dataclass(frozen=True)
class RaySegmentBundle:
    """Finite in-object ray segments for volumetric source deposition.

    Each row is a closed chord ``starts[i] → ends[i]`` with ballistic intensity
    ``intensities[i]``. ``ray_ids`` carry parent source-ray lineage;
    ``segment_ids`` and ``path_order`` preserve stable segment identity within
    transport paths.

    Parameters
    ----------
    starts, ends:
        Segment endpoints, shape ``[N, 3]`` (mesh units).
    intensities:
        Ballistic intensity at segment start, shape ``[N]``.
    ray_ids:
        Parent source-ray identifier per segment, shape ``[N]``.
    segment_ids:
        Optional stable segment ids; defaults to row indices ``0..N-1``.
    path_order:
        Zero-based order within each ``ray_id`` path; inferred from row order
        when omitted.
    """

    starts: np.ndarray
    ends: np.ndarray
    intensities: np.ndarray
    ray_ids: np.ndarray
    segment_ids: np.ndarray | None = None
    path_order: np.ndarray | None = None

    def __post_init__(self) -> None:
        starts = np.asarray(self.starts, dtype=float)
        ends = np.asarray(self.ends, dtype=float)
        intensities = np.asarray(self.intensities, dtype=float)
        ray_ids = np.asarray(self.ray_ids, dtype=int)
        if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 3:
            raise ValueError("starts/ends must share shape [N, 3]")
        n = len(starts)
        if intensities.ndim != 1 or len(intensities) != n:
            raise ValueError("intensities must be shape [N]")
        if ray_ids.ndim != 1 or len(ray_ids) != n:
            raise ValueError("ray_ids must be shape [N]")
        if np.any(~np.isfinite(intensities)) or np.any(intensities < 0):
            raise ValueError("intensities must be finite and non-negative")
        if self.segment_ids is None:
            segment_ids = np.arange(n, dtype=int)
        else:
            segment_ids = np.asarray(self.segment_ids, dtype=int)
            if segment_ids.ndim != 1 or len(segment_ids) != n:
                raise ValueError("segment_ids must be shape [N]")
        if self.path_order is None:
            path_order = np.empty(n, dtype=int)
            next_order: dict[int, int] = {}
            for i, path_id in enumerate(ray_ids.tolist()):
                order = next_order.get(path_id, 0)
                path_order[i] = order
                next_order[path_id] = order + 1
        else:
            path_order = np.asarray(self.path_order, dtype=int)
            if path_order.ndim != 1 or len(path_order) != n:
                raise ValueError("path_order must be shape [N]")
            for path_id in np.unique(ray_ids):
                orders = path_order[ray_ids == path_id]
                expected = np.arange(len(orders), dtype=int)
                if not np.array_equal(np.sort(orders), expected):
                    raise ValueError(
                        "path_order must be unique and contiguous from zero within "
                        f"transport path {int(path_id)}"
                    )
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "ends", ends)
        object.__setattr__(self, "intensities", intensities)
        object.__setattr__(self, "ray_ids", ray_ids)
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "path_order", path_order)

    @property
    def n_segments(self) -> int:
        """Number of segment rows in the bundle."""
        return len(self.starts)

    @property
    def n_rays(self) -> int:
        """Alias for :attr:`n_segments` (segment row count)."""
        return self.n_segments

    @property
    def n_parent_rays(self) -> int:
        """Number of distinct ``ray_ids`` values."""
        return int(len(np.unique(self.ray_ids)))

    @property
    def n_transport_paths(self) -> int:
        """Number of distinct propagated transport paths (same as ``n_parent_rays``)."""
        return self.n_parent_rays

    def subset(self, mask_or_indices, intensity_scale=1.0):
        """Return a bundle containing only selected segment rows.

        Parameters
        ----------
        mask_or_indices:
            Boolean mask, integer indices, slice, or list selecting rows.
        intensity_scale:
            Multiplier applied to selected segment intensities (e.g. path
            subsampling correction).

        Notes
        -----
        ``ray_ids``, ``segment_ids``, and ``path_order`` are copied unchanged to
        preserve lineage; they are not renumbered.
        """
        selector = mask_or_indices

        starts = np.asarray(self.starts)
        n_segments = len(starts)

        if isinstance(selector, slice):
            pass
        else:
            selector = np.asarray(selector)

            if selector.dtype == bool:
                if selector.shape[0] != n_segments:
                    raise ValueError(
                        "Boolean segment mask has length "
                        + str(selector.shape[0])
                        + ", expected "
                        + str(n_segments)
                        + "."
                    )
            else:
                selector = selector.astype(int)

        return type(self)(
            starts=np.asarray(self.starts)[selector].copy(),
            ends=np.asarray(self.ends)[selector].copy(),
            intensities=(
                np.asarray(self.intensities, dtype=float)[selector].copy()
                * float(intensity_scale)
            ),
            ray_ids=np.asarray(self.ray_ids)[selector].copy(),
            segment_ids=np.asarray(self.segment_ids)[selector].copy(),
            path_order=np.asarray(self.path_order)[selector].copy(),
        )


def make_synthetic_axis_ray(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    intensity: float = 1.0,
    *,
    ray_id: int = 0,
) -> RaySegmentBundle:
    """Build a one-segment :class:`RaySegmentBundle` for tests and fixtures.

    Parameters
    ----------
    start, end:
        Segment endpoints in mesh coordinates.
    intensity:
        Ballistic intensity at ``start``.
    ray_id:
        Parent ray identifier stored in ``ray_ids``.

    Returns
    -------
    RaySegmentBundle
        Single-row segment bundle.
    """
    return RaySegmentBundle(
        starts=np.asarray([start], dtype=float),
        ends=np.asarray([end], dtype=float),
        intensities=np.asarray([intensity], dtype=float),
        ray_ids=np.asarray([int(ray_id)], dtype=int),
    )


def in_object_segments_from_rays(
    mesh: trimesh.Trimesh,
    rays: Any,
    *,
    parent_ray_ids: np.ndarray | Sequence[int] | None = None,
) -> RaySegmentBundle:
    """Close internal refracted rays into finite segments ending at exit hits.

    :func:`deposit_ray_source` requires finite ``(starts, ends, intensities)``
    chords, not open ``(origins, directions, weights)`` rays. Typical pipeline:

    1. ``refract_ray_bundle(...)`` → internal ``SourceRayBundle``
    2. this helper (exit hit closes each segment)
    3. ``deposit_ray_source(diffusion_mesh, segments, ...)``

    Intensities are taken from ``rays.weights``, else ``rays.intensities``, else
    ones. ``parent_ray_ids`` maps each internal ray row to its parent source-ray
    index (e.g. ``refract_result.parent_indices``); defaults to internal row index.

    Parameters
    ----------
    mesh:
        Surface mesh for exit visibility.
    rays:
        Internal ray bundle with ``origins`` / ``directions``.
    parent_ray_ids:
        Optional lineage array, shape ``[n_rays]``.

    Returns
    -------
    RaySegmentBundle
        One segment per ray with a valid exit hit; empty when none hit.
    """
    origins = np.asarray(rays.origins, dtype=float)
    directions = np.asarray(rays.directions, dtype=float)
    if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("rays.origins/directions must share shape [N, 3]")

    n_rays = len(origins)
    intensities = getattr(rays, "weights", None)
    if intensities is None:
        intensities = getattr(rays, "intensities", None)
    if intensities is None:
        intensities = np.ones(n_rays, dtype=float)
    else:
        intensities = np.asarray(intensities, dtype=float)
        if intensities.shape != (n_rays,):
            raise ValueError(
                f"rays weights/intensities must be [{n_rays}], got {intensities.shape}"
            )

    hit_valid, _depth, hit_faces, hit_points = first_visible_hits_with_points(mesh, rays)
    active = hit_valid & (hit_faces >= 0)
    parent = np.flatnonzero(active)
    if parent.size == 0:
        return RaySegmentBundle(
            starts=np.zeros((0, 3), dtype=float),
            ends=np.zeros((0, 3), dtype=float),
            intensities=np.zeros(0, dtype=float),
            ray_ids=np.zeros(0, dtype=int),
        )

    if parent_ray_ids is not None:
        parent_ray_ids_arr = np.asarray(parent_ray_ids, dtype=int)
        if parent_ray_ids_arr.shape != (n_rays,):
            raise ValueError(
                f"parent_ray_ids must be shape [{n_rays}], got {parent_ray_ids_arr.shape}"
            )
        lineage = parent_ray_ids_arr[parent]
    else:
        lineage = parent.astype(int)

    return RaySegmentBundle(
        starts=origins[parent],
        ends=np.asarray(hit_points[parent], dtype=float),
        intensities=intensities[parent],
        ray_ids=lineage,
        segment_ids=parent.astype(int),
    )


def _as_ray_segment_bundle(rays: Any) -> RaySegmentBundle:
    if isinstance(rays, RaySegmentBundle):
        return rays
    if (
        hasattr(rays, "starts")
        and hasattr(rays, "ends")
        and hasattr(rays, "intensities")
    ):
        ray_ids = getattr(rays, "ray_ids", None)
        if ray_ids is None:
            n = len(np.asarray(rays.starts))
            ray_ids = np.arange(n, dtype=int)
        return RaySegmentBundle(
            starts=rays.starts,
            ends=rays.ends,
            intensities=rays.intensities,
            ray_ids=ray_ids,
            segment_ids=getattr(rays, "segment_ids", None),
            path_order=getattr(rays, "path_order", None),
        )
    raise TypeError(
        "deposit_ray_source expects a RaySegmentBundle with "
        "starts/ends/intensities/ray_ids. SourceRayBundle (origins/directions/weights) "
        "must be closed first, e.g. "
        "segments = in_object_segments_from_rays(surface_mesh, internal_rays)."
    )


def _ray_tet_interval(
    p0: np.ndarray,
    direction: np.ndarray,
    length: float,
    tet_vertices: np.ndarray,
    eps: float = 1e-12,
) -> tuple[bool, float, float]:
    """Intersect a finite ray segment with a tetrahedron.

    Ray is parameterised as:

        p(t) = p0 + t * direction

    with direction assumed normalised and t in [0, length].

    Returns
    -------
    hit, t_enter, t_exit
        If hit is True, the ray segment intersects the tetrahedron over
        [t_enter, t_exit].
    """
    v = np.asarray(tet_vertices, dtype=float)

    faces = (
        (0, 1, 2, 3),
        (0, 3, 1, 2),
        (0, 2, 3, 1),
        (1, 3, 2, 0),
    )

    t_enter = 0.0
    t_exit = float(length)

    for i, j, k, opposite in faces:
        a = v[i]
        b = v[j]
        c = v[k]
        opp = v[opposite]

        n = np.cross(b - a, c - a)

        norm_n = float(np.linalg.norm(n))
        if norm_n <= eps:
            return False, 0.0, 0.0

        if float(np.dot(n, opp - a)) > 0.0:
            n = -n

        numer = float(np.dot(n, p0 - a))
        denom = float(np.dot(n, direction))

        if abs(denom) <= eps:
            if numer > eps:
                return False, 0.0, 0.0
            continue

        t_hit = -numer / denom

        if denom > 0.0:
            t_exit = min(t_exit, t_hit)
        else:
            t_enter = max(t_enter, t_hit)

        if t_exit <= t_enter + eps:
            return False, 0.0, 0.0

    t_enter = max(0.0, t_enter)
    t_exit = min(float(length), t_exit)

    if t_exit <= t_enter + eps:
        return False, 0.0, 0.0

    return True, float(t_enter), float(t_exit)


def _ray_tet_intersections(
    diffusion_mesh: DiffusionMesh,
    p0: np.ndarray,
    direction: np.ndarray,
    length: float,
    eps: float = 1e-12,
) -> list[tuple[int, float, float]]:
    """Return exact ray/tetrahedron intersections sorted along the ray.

    Returns
    -------
    hits
        List of tuples:

            (tet_index, t_enter, t_exit)

        where t is measured from p0 along the normalised ray direction.
    """
    nodes = np.asarray(diffusion_mesh.nodes, dtype=float)
    tets = np.asarray(diffusion_mesh.tets, dtype=int)

    hits: list[tuple[int, float, float]] = []

    for e, tet in enumerate(tets):
        tet_vertices = nodes[tet]

        hit, t_enter, t_exit = _ray_tet_interval(
            p0=p0,
            direction=direction,
            length=length,
            tet_vertices=tet_vertices,
            eps=eps,
        )

        if not hit:
            continue

        if t_exit <= t_enter + eps:
            continue

        hits.append((e, t_enter, t_exit))

    hits.sort(key=lambda item: item[1])
    return hits


def distribute_energy_along_segment(
    diffusion_mesh: DiffusionMesh,
    start: np.ndarray,
    end: np.ndarray,
    energy: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Split fixed energy across tets intersected by a segment (no Beer–Lambert).

    Unlike :func:`deposit_ray_source`, this does not apply attenuation along the
    chord. The caller supplies already-integrated energy (e.g. particle-scatter
    ``E_scat``), which is partitioned by exact in-tet path length:

        E[e] += energy * dl[e] / sum(dl)

    Uses the same exact ray–tetrahedron interval intersection as deposition.

    Parameters
    ----------
    diffusion_mesh:
        Coarse tet mesh for localization.
    start, end:
        Segment endpoints, shape ``(3,)`` (mesh units).
    energy:
        Total energy to distribute (non-negative).
    eps:
        Numerical tolerance for degenerate intersections.

    Returns
    -------
    np.ndarray, shape ``[n_tets]``
        Per-tet deposited energy; all zeros when the segment misses or has zero
        length.

    Raises
    ------
    ValueError
        If ``energy < 0``.

    Notebook / protocol:
        M5 particle-scatter deposition helper; caller owns fallback when sum(dl)=0.
    """
    energy_v = float(energy)
    out = np.zeros(diffusion_mesh.n_tets, dtype=float)
    if energy_v == 0.0:
        return out
    if energy_v < 0.0:
        raise ValueError("energy must be non-negative")

    p0 = np.asarray(start, dtype=float).reshape(3)
    p1 = np.asarray(end, dtype=float).reshape(3)
    segment = p1 - p0
    length = float(np.linalg.norm(segment))
    if length <= eps:
        return out

    direction = segment / length
    hits = _ray_tet_intersections(
        diffusion_mesh=diffusion_mesh,
        p0=p0,
        direction=direction,
        length=length,
        eps=eps,
    )
    if not hits:
        return out

    lengths = np.asarray([float(t_exit - t_enter) for _, t_enter, t_exit in hits], dtype=float)
    total_length = float(np.sum(lengths))
    if total_length <= eps:
        return out

    for (tet_index, _t_enter, _t_exit), dl in zip(hits, lengths):
        if dl <= 0.0:
            continue
        out[int(tet_index)] += energy_v * (dl / total_length)
    return out


def deposit_ray_source(
    diffusion_mesh: DiffusionMesh,
    rays: RaySegmentBundle | Any,
    material: OpticalMaterialConfig | None = None,
    *,
    mu_s: float | None = None,
    mu_a: float | None = None,
) -> SourceDepositionResult:
    """Deposit ballistic ray segments into volumetric scattering source density ``S(x)``.

    For each finite segment, intersects the coarse tet mesh with exact ray–tet
    intervals, sorts hits along the chord, and applies Beer–Lambert attenuation
    *only inside intersected tets* (empty space between tets does not attenuate).
    Scattered energy accumulates per tet; absorption is loss. Element density is

        S[e] = E_scat_elem[e] / volume[e]

    ``rays`` must be a :class:`RaySegmentBundle`. Convert refracted internal
    :class:`~gummybear.rays.source.SourceRayBundle` rays with
    :func:`in_object_segments_from_rays` first.

    Parameters
    ----------
    diffusion_mesh:
        Coarse :class:`DiffusionMesh` from Netgen meshing.
    rays:
        Finite segment bundle or compatible object with starts/ends/intensities.
    material:
        Supplies scattering coefficient ``mu_s`` / absorption coefficient ``mu_a`` when coefficient overrides are omitted.
    mu_s, mu_a:
        Optional coefficient overrides (1 / mesh units).

    Returns
    -------
    SourceDepositionResult

    Raises
    ------
    ValueError
        Missing material/coefficients or negative ``mu_s`` / ``mu_a``.
    TypeError
        When ``rays`` is not a segment bundle.

    Notebook / protocol:
        M4B exact ray–tet deposition mainline.

    See also:
        :func:`~gummybear.optics.diffusion_solve.solve_diffusion` — consumes ``S_clean`` per tet.
        :func:`~gummybear.particles.perturbation.compute_transport_source_correction` — particle perturbation of ``S``.
    """
    rays = _as_ray_segment_bundle(rays)

    if material is not None:
        mu_s_v = float(material.mu_scatter if mu_s is None else mu_s)
        mu_a_v = float(material.mu_absorption if mu_a is None else mu_a)
    else:
        if mu_s is None or mu_a is None:
            raise ValueError("Provide material or both mu_s and mu_a")
        mu_s_v = float(mu_s)
        mu_a_v = float(mu_a)

    if mu_s_v < 0.0 or mu_a_v < 0.0:
        raise ValueError("mu_s and mu_a must be non-negative")

    mu_total = mu_s_v + mu_a_v

    n_tets = diffusion_mesh.n_tets
    E_scat = np.zeros(n_tets, dtype=float)

    total_absorbed = 0.0
    remaining_direct = 0.0
    total_input = float(np.sum(rays.intensities))

    volumes = np.asarray(diffusion_mesh.volumes, dtype=float)

    for i in range(rays.n_rays):
        p0 = np.asarray(rays.starts[i], dtype=float)
        p1 = np.asarray(rays.ends[i], dtype=float)
        I0 = float(rays.intensities[i])

        segment = p1 - p0
        length = float(np.linalg.norm(segment))

        if length <= 0.0 or I0 <= 0.0:
            remaining_direct += I0
            continue

        direction = segment / length

        hits = _ray_tet_intersections(
            diffusion_mesh=diffusion_mesh,
            p0=p0,
            direction=direction,
            length=length,
        )

        if not hits:
            remaining_direct += I0
            continue

        I_local = I0

        for e, t_enter, t_exit in hits:
            dl = float(t_exit - t_enter)

            if dl <= 0.0:
                continue

            if mu_total > 0.0:
                attenuation = float(np.exp(-mu_total * dl))
                total_loss = I_local * (1.0 - attenuation)

                dE_scat = total_loss * (mu_s_v / mu_total)
                dE_abs = total_loss * (mu_a_v / mu_total)

                I_local = I_local * attenuation
            else:
                dE_scat = 0.0
                dE_abs = 0.0

            E_scat[e] += dE_scat
            total_absorbed += dE_abs

        remaining_direct += I_local

    S = np.zeros(n_tets, dtype=float)
    np.divide(E_scat, volumes, out=S, where=volumes > 0)

    return SourceDepositionResult(
        S_clean=S,
        E_scat_elem=E_scat,
        total_ballistic_input=total_input,
        total_scattered=float(np.sum(E_scat)),
        total_absorbed=float(total_absorbed),
        remaining_direct_energy=float(remaining_direct),
        mu_s=mu_s_v,
        mu_a=mu_a_v,
    )
