"""Sample volumetric fluence field ``Phi`` at camera hit points to form ``I_diffuse``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .diffusion_mesh import DiffusionMesh
from .diffusion_solve import sample_phi_at_points

# sample_mode codes for cached Phi sampling localization
SAMPLE_MODE_INVALID = 0
SAMPLE_MODE_BARYCENTRIC = 1
SAMPLE_MODE_MEAN_NODAL = 2


@dataclass(frozen=True)
class DiffuseSampleResult:
    """Diffuse camera channel built from fluence field ``Phi`` samples at hit points.

    Attributes
    ----------
    I_diffuse:
        Diffuse intensity image, shape ``sample_shape`` (typically ``[H, W]``).
    Phi_at_hits:
        Flat ``Phi`` samples at hits, shape ``[H*W]``; invalid pixels are zero
        in ``I_diffuse`` but may be NaN in this array before masking.
    sample_shape:
        ``(H, W)`` camera grid shape.
    """

    I_diffuse: np.ndarray  # same shape as camera mask / image
    Phi_at_hits: np.ndarray  # flat [N] samples (NaN for misses)
    sample_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class PhiSamplingLocalization:
    """Geometry-only tet localization for applying nodal fluence field ``Phi`` at query points.

    Computed once from hit points and reused across multiple ``Phi_nodes`` fields
    (e.g. clean vs perturbed solves). Independent of fluence values.

    Attributes
    ----------
    sample_mode:
        Per-point mode code (``SAMPLE_MODE_BARYCENTRIC`` or
        ``SAMPLE_MODE_MEAN_NODAL``), shape ``[N]``, dtype ``int8``.
    tet_id:
        Containing or nearest tet index; ``-1`` when invalid, shape ``[N]``.
    barycentric:
        Barycentric weights for nodal interpolation, shape ``[N, 4]``.
    """

    sample_mode: np.ndarray  # int8 [N]
    tet_id: np.ndarray  # int64 [N]; -1 when invalid
    barycentric: np.ndarray  # float64 [N, 4]


def localize_points_in_diffusion_mesh(
    diffusion_mesh: DiffusionMesh,
    points: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    barycentric_tolerance: float = 1e-10,
) -> PhiSamplingLocalization:
    """Locate containing (or nearest) tetrahedra for 3D query points.

    Searches nearby tets by centroid KD-tree, then all tets if needed. Surface
    hits slightly outside the volume use barycentric extrapolation in the nearest
    valid tet; degenerate meshes fall back to mean-nodal sampling. Does not
    read ``Phi_nodes`` — pair with :func:`apply_phi_localization`.

    Parameters
    ----------
    diffusion_mesh:
        Coarse tet mesh.
    points:
        Query locations, shape ``[N, 3]`` (mesh units).
    valid_mask:
        Optional boolean mask, shape ``[N]``; non-finite points are excluded.
    barycentric_tolerance:
        Inclusive tolerance on barycentric coordinates for "inside" tests.

    Returns
    -------
    PhiSamplingLocalization
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N, 3]")
    if barycentric_tolerance < 0:
        raise ValueError("barycentric_tolerance must be nonnegative")

    n_points = len(points)
    sample_mode = np.zeros(n_points, dtype=np.int8)
    tet_id = np.full(n_points, -1, dtype=np.int64)
    barycentric = np.zeros((n_points, 4), dtype=float)

    if valid_mask is None:
        valid = np.all(np.isfinite(points), axis=1)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (n_points,):
            raise ValueError("valid_mask must have shape [N]")
        valid = valid & np.all(np.isfinite(points), axis=1)

    if not np.any(valid) or diffusion_mesh.n_tets == 0:
        return PhiSamplingLocalization(
            sample_mode=sample_mode,
            tet_id=tet_id,
            barycentric=barycentric,
        )

    tets = diffusion_mesh.tets
    vertices = diffusion_mesh.nodes[tets]
    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]
    edge_3 = vertices[:, 3] - vertices[:, 0]
    determinants = np.einsum(
        "ti,ti->t", edge_1, np.cross(edge_2, edge_3)
    )
    invertible = np.abs(determinants) > np.finfo(float).eps

    tree = cKDTree(diffusion_mesh.centroids)
    candidate_count = min(32, diffusion_mesh.n_tets)

    def barycentric_weights(point: np.ndarray, tet_ids: np.ndarray) -> np.ndarray:
        relative = point - vertices[tet_ids, 0]
        denominator = determinants[tet_ids]
        weight_1 = np.einsum(
            "ti,ti->t",
            relative,
            np.cross(edge_2[tet_ids], edge_3[tet_ids]),
        ) / denominator
        weight_2 = np.einsum(
            "ti,ti->t",
            edge_1[tet_ids],
            np.cross(relative, edge_3[tet_ids]),
        ) / denominator
        weight_3 = np.einsum(
            "ti,ti->t",
            edge_1[tet_ids],
            np.cross(edge_2[tet_ids], relative),
        ) / denominator
        return np.column_stack(
            (
                1.0 - weight_1 - weight_2 - weight_3,
                weight_1,
                weight_2,
                weight_3,
            )
        )

    for point_index in np.flatnonzero(valid):
        point = points[point_index]
        _, nearby = tree.query(point, k=candidate_count)
        candidate_ids = np.atleast_1d(nearby).astype(int)
        candidate_ids = candidate_ids[invertible[candidate_ids]]
        weights = barycentric_weights(point, candidate_ids)
        inside = np.all(
            (weights >= -barycentric_tolerance)
            & (weights <= 1.0 + barycentric_tolerance),
            axis=1,
        )

        if np.any(inside):
            match = int(np.flatnonzero(inside)[0])
            tet_id[point_index] = int(candidate_ids[match])
            barycentric[point_index] = weights[match]
            sample_mode[point_index] = SAMPLE_MODE_BARYCENTRIC
            continue

        all_ids = np.flatnonzero(invertible)
        all_weights = barycentric_weights(point, all_ids)
        inside = np.all(
            (all_weights >= -barycentric_tolerance)
            & (all_weights <= 1.0 + barycentric_tolerance),
            axis=1,
        )
        if np.any(inside):
            match = int(np.flatnonzero(inside)[0])
            tet_id[point_index] = int(all_ids[match])
            barycentric[point_index] = all_weights[match]
            sample_mode[point_index] = SAMPLE_MODE_BARYCENTRIC
            continue

        nearest_order = np.atleast_1d(
            tree.query(point, k=diffusion_mesh.n_tets)[1]
        ).astype(int)
        valid_nearest = nearest_order[invertible[nearest_order]]
        if len(valid_nearest) == 0:
            tet_id[point_index] = int(nearest_order[0])
            sample_mode[point_index] = SAMPLE_MODE_MEAN_NODAL
            continue
        chosen = int(valid_nearest[0])
        tet_id[point_index] = chosen
        barycentric[point_index] = barycentric_weights(
            point, np.asarray([chosen])
        )[0]
        sample_mode[point_index] = SAMPLE_MODE_BARYCENTRIC

    return PhiSamplingLocalization(
        sample_mode=sample_mode,
        tet_id=tet_id,
        barycentric=barycentric,
    )


def apply_phi_localization(
    diffusion_mesh: DiffusionMesh,
    Phi_nodes: np.ndarray,
    localization: PhiSamplingLocalization,
) -> np.ndarray:
    """Evaluate nodal fluence field ``Phi`` at pre-localized points via barycentric weighted sums.

    Barycentric mode: ``Phi(p) = sum_k w_k * Phi_nodes[tet_k]``.
    Mean-nodal fallback: average of the four corner nodes. Invalid modes stay NaN.

    Parameters
    ----------
    diffusion_mesh:
        Coarse tet mesh.
    Phi_nodes:
        Nodal fluence field ``Phi``, shape ``[n_nodes]``.
    localization:
        Precomputed :class:`PhiSamplingLocalization` from
        :func:`localize_points_in_diffusion_mesh`.

    Returns
    -------
    np.ndarray, shape ``[N]``
        Interpolated fluence per query point.
    """
    Phi_nodes = np.asarray(Phi_nodes, dtype=float)
    if Phi_nodes.shape != (diffusion_mesh.n_nodes,):
        raise ValueError("Phi_nodes length must match diffusion_mesh.n_nodes")

    sample_mode = np.asarray(localization.sample_mode)
    tet_ids = np.asarray(localization.tet_id, dtype=np.int64)
    bary = np.asarray(localization.barycentric, dtype=float)
    n_points = len(sample_mode)
    if tet_ids.shape != (n_points,) or bary.shape != (n_points, 4):
        raise ValueError("localization arrays must agree on length N")

    out = np.full(n_points, np.nan, dtype=float)
    tets = diffusion_mesh.tets

    bary_mask = sample_mode == SAMPLE_MODE_BARYCENTRIC
    if np.any(bary_mask):
        ids = tet_ids[bary_mask]
        out[bary_mask] = np.einsum(
            "ij,ij->i",
            bary[bary_mask],
            Phi_nodes[tets[ids]],
        )

    mean_mask = sample_mode == SAMPLE_MODE_MEAN_NODAL
    if np.any(mean_mask):
        ids = tet_ids[mean_mask]
        out[mean_mask] = np.mean(Phi_nodes[tets[ids]], axis=1)

    return out


def interpolate_phi_nodes_to_points(
    diffusion_mesh: DiffusionMesh,
    Phi_nodes: np.ndarray,
    points: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    barycentric_tolerance: float = 1e-10,
) -> np.ndarray:
    """Interpolate nodal fluence field ``Phi`` at 3D points using tetrahedral barycentric weights.

    Combines :func:`localize_points_in_diffusion_mesh` and
    :func:`apply_phi_localization`. Invalid or non-finite points return NaN.

    Parameters
    ----------
    diffusion_mesh:
        Coarse tet mesh.
    Phi_nodes:
        Nodal fluence field ``Phi``, shape ``[n_nodes]``.
    points:
        Query locations, shape ``[N, 3]``.
    valid_mask:
        Optional boolean mask, shape ``[N]``.
    barycentric_tolerance:
        Passed to localization.

    Returns
    -------
    np.ndarray, shape ``[N]``
    """
    Phi_nodes = np.asarray(Phi_nodes, dtype=float)
    if Phi_nodes.shape != (diffusion_mesh.n_nodes,):
        raise ValueError("Phi_nodes length must match diffusion_mesh.n_nodes")
    localization = localize_points_in_diffusion_mesh(
        diffusion_mesh,
        points,
        valid_mask=valid_mask,
        barycentric_tolerance=barycentric_tolerance,
    )
    return apply_phi_localization(diffusion_mesh, Phi_nodes, localization)


def sample_phi_at_hit_points(
    diffusion_mesh: DiffusionMesh,
    Phi_nodes: np.ndarray,
    hit_points: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    interpolate: bool = True,
    localization: PhiSamplingLocalization | None = None,
) -> np.ndarray:
    """Sample fluence field ``Phi`` at camera hit points for diffuse image formation.

    Requires 3D ``hit_points`` from a camera visibility pass (e.g.
    ``first_visible_hits_with_points``). By default uses barycentric tet
    interpolation; ``interpolate=False`` selects legacy nearest-tet mean nodal
    sampling via :func:`~gummybear.optics.diffusion_solve.sample_phi_at_points`.
    When ``localization`` is supplied with ``interpolate=True``, tet search is
    skipped.

    Valid hits return finite samples with NaN replaced by ``0.0``; invalid hits
    are zero when ``valid_mask`` is provided.

    Parameters
    ----------
    diffusion_mesh:
        Coarse tet mesh.
    Phi_nodes:
        Nodal fluence from :func:`~gummybear.optics.diffusion_solve.solve_diffusion`.
    hit_points:
        Camera hit locations, shape ``[N, 3]``.
    valid_mask:
        Optional boolean mask, shape ``[N]``.
    interpolate:
        Use barycentric interpolation (default) or legacy nearest-tet mean.
    localization:
        Optional precomputed localization cache.

    Returns
    -------
    np.ndarray, shape ``[N]``
        Fluence samples for diffuse channel assembly.

    Notebook / protocol:
        M4D camera fluence sampling; default barycentric, legacy via ``interpolate=False``.

    See also:
        :func:`first_visible_hits_with_points` — typical source of ``hit_points``.
        :func:`sample_diffuse_image` — reshape samples into ``I_diffuse``.
    """
    hit_points = np.asarray(hit_points, dtype=float)
    if hit_points.ndim != 2 or hit_points.shape[1] != 3:
        raise ValueError("hit_points must be [N, 3]")

    if interpolate:
        if localization is None:
            Phi_at = interpolate_phi_nodes_to_points(
                diffusion_mesh, Phi_nodes, hit_points, valid_mask=valid_mask
            )
        else:
            Phi_at = apply_phi_localization(
                diffusion_mesh, Phi_nodes, localization
            )
    else:
        Phi_at = sample_phi_at_points(diffusion_mesh, Phi_nodes, hit_points)
    if valid_mask is None:
        valid_mask = np.all(np.isfinite(hit_points), axis=1)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (len(hit_points),):
            raise ValueError("valid_mask must have shape [N]")

    out = np.zeros(len(hit_points), dtype=float)
    out[valid_mask] = np.nan_to_num(Phi_at[valid_mask], nan=0.0)
    return out


def sample_diffuse_image(
    diffusion_mesh: DiffusionMesh,
    Phi_nodes: np.ndarray,
    hit_points: np.ndarray,
    valid_mask: np.ndarray,
    sample_shape: tuple[int, int],
    *,
    exitance_scale: float = 1.0,
    interpolate: bool = True,
    localization: PhiSamplingLocalization | None = None,
) -> DiffuseSampleResult:
    """Build ``I_diffuse`` by sampling fluence field ``Phi`` at flattened camera hit points.

    Requires ``hit_points`` and ``valid_mask`` from the same camera visibility
    pass, flattened to ``H*W``. Applies ``exitance_scale`` as an explicit
    provisional scale (not a physical BRDF — record in run metadata). Invalid
    pixels are exactly zero. Pass ``localization`` to reuse tet maps across
    multiple ``Phi_nodes`` fields.

    Parameters
    ----------
    diffusion_mesh:
        Coarse tet mesh.
    Phi_nodes:
        Nodal fluence field ``Phi``, shape ``[n_nodes]``.
    hit_points:
        Flat hit locations, shape ``[H*W, 3]``.
    valid_mask:
        Flat boolean validity, shape ``[H*W]``.
    sample_shape:
        ``(H, W)`` image shape.
    exitance_scale:
        Multiplier on sampled fluence (explicit relative scale).
    interpolate:
        Barycentric (default) vs legacy nearest-tet mean sampling.
    localization:
        Optional precomputed :class:`PhiSamplingLocalization`.

    Returns
    -------
    DiffuseSampleResult

    Notebook / protocol:
        M4D diffuse camera channel; pairs with hybrid compose for ``I_total``.

    See also:
        :func:`~gummybear.optics.hybrid_compose.compose_hybrid_image` — ``I_total = alpha * I_direct + I_diffuse``.
    """
    H, W = sample_shape
    expected = H * W
    hit_points = np.asarray(hit_points, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if hit_points.shape != (expected, 3):
        raise ValueError(
            f"hit_points must be [{expected}, 3] for sample_shape={sample_shape}"
        )
    if valid_mask.shape != (expected,):
        raise ValueError(f"valid_mask must be [{expected}]")

    Phi_at = sample_phi_at_hit_points(
        diffusion_mesh,
        Phi_nodes,
        hit_points,
        valid_mask=valid_mask,
        interpolate=interpolate,
        localization=localization,
    )
    I_flat = float(exitance_scale) * Phi_at
    I_flat[~valid_mask] = 0.0
    I_img = I_flat.reshape(H, W)
    return DiffuseSampleResult(
        I_diffuse=I_img,
        Phi_at_hits=Phi_at,
        sample_shape=sample_shape,
    )
