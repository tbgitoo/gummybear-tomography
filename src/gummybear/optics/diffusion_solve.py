"""Homogeneous optical diffusion finite-element method (FEM) solve on a coarse DiffusionMesh (NGSolve)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .diffusion_mesh import DiffusionMesh


ROBIN_BOUNDARY_MODEL = "effective_refractive_boundary"
"""Boundary model identifier for Robin boundary condition (extrapolated-flux leakage) in :func:`solve_diffusion`."""


@dataclass(frozen=True)
class DiffusionSolveResult:
    """Steady-state fluence field ``Phi`` solution and finite-element method (FEM) operator metadata.

    Attributes
    ----------
    Phi_nodes:
        H1 nodal fluence field ``Phi`` (volumetric light density from the diffusion solve), shape ``[n_nodes]``.
    Phi_tets:
        Mean nodal ``Phi`` per tet (diagnostic), shape ``[n_tets]``.
    residual_norm:
        Relative linear-system residual ``||A u - f|| / ||f||``.
    operator_cache_key:
        Hashable dict of mesh and operator parameters.
    robin_boundary_model:
        Boundary condition (BC) identifier (see :data:`ROBIN_BOUNDARY_MODEL`).
    extrapolation_length:
        Robin boundary condition length scale ``L`` in ``(D/L) u v`` boundary term (mesh units).
    D, mu_a:
        Diffusion coefficient and absorption coefficient ``mu_a`` used in the solve.
    diagnostics:
        DOF counts, NGSolve version, operator key hash, etc.
    """

    Phi_nodes: np.ndarray  # [n_nodes] H1 nodal fluence
    Phi_tets: np.ndarray  # [n_tets] mean nodal Phi per tet (diagnostic)
    residual_norm: float | None
    operator_cache_key: dict[str, Any]
    robin_boundary_model: str = ROBIN_BOUNDARY_MODEL
    extrapolation_length: float = 1.0
    D: float = 1.0
    mu_a: float = 0.1
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _require_ngsolve():
    try:
        import ngsolve
    except ImportError as exc:
        raise ImportError(
            "M4C requires the optional 'fem' extra (pip install '.[fem]')."
        ) from exc
    return ngsolve


def default_diffusion_coefficient(mu_s: float, mu_a: float) -> float:
    """Return optically thick diffusion coefficient ``D ≈ 1 / (3 μ_total)``.

    Parameters
    ----------
    mu_s, mu_a:
        Scattering coefficient ``mu_s`` and absorption coefficient ``mu_a`` (1 / mesh units).

    Returns
    -------
    float
        Diffusion coefficient ``D``.

    Raises
    ------
    ValueError
        When ``mu_s + mu_a <= 0``.
    """
    mu_t = float(mu_s) + float(mu_a)
    if mu_t <= 0.0:
        raise ValueError("mu_total must be positive to define D")
    return 1.0 / (3.0 * mu_t)


def operator_cache_keys(
    diffusion_mesh: DiffusionMesh,
    *,
    D: float,
    mu_a: float,
    extrapolation_length: float,
    robin_boundary_model: str = ROBIN_BOUNDARY_MODEL,
    fem_order: int = 1,
) -> dict[str, Any]:
    """Build a hashable key dict for caching diffusion operators and solutions.

    Parameters
    ----------
    diffusion_mesh:
        Mesh whose :meth:`DiffusionMesh.content_hash` enters the key.
    D, mu_a:
        Diffusion coefficient and absorption coefficient ``mu_a`` (1 / mesh units).
    extrapolation_length:
        Robin boundary condition length scale ``L`` (mesh units).
    robin_boundary_model:
        Boundary model name (default :data:`ROBIN_BOUNDARY_MODEL`).
    fem_order:
        H1 finite-element method (FEM) polynomial order.

    Returns
    -------
    dict[str, Any]
        Serializable operator cache key fields.
    """
    return {
        "diffusion_mesh_hash": diffusion_mesh.content_hash(),
        "D": float(D),
        "mu_a": float(mu_a),
        "robin_boundary_model": robin_boundary_model,
        "extrapolation_length": float(extrapolation_length),
        "fem_order": int(fem_order),
    }


def _reindex_phi_to_diffusion_nodes(ngmesh, nodes: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Match NGSolve vertex order to DiffusionMesh.nodes by coordinates.

    NGSolve may expose vertices in a different order than the project-level
    DiffusionMesh object, so coordinate matching restores the expected indexing.
    """
    ng_coords = np.array([list(v.point) for v in ngmesh.vertices], dtype=float)
    if len(ng_coords) != len(phi):
        raise RuntimeError("ngmesh vertices and Phi length mismatch")
    out = np.zeros(len(nodes), dtype=float)
    for i, p in enumerate(nodes):
        d2 = np.sum((ng_coords - p) ** 2, axis=1)
        out[i] = phi[int(np.argmin(d2))]
    return out


def solve_diffusion(
    diffusion_mesh: DiffusionMesh,
    S_clean: np.ndarray,
    *,
    D: float,
    mu_a: float,
    extrapolation_length: float = 1.0,
    robin_boundary_model: str = ROBIN_BOUNDARY_MODEL,
    fem_order: int = 1,
    use_dirichlet_zero_debug: bool = False,
) -> DiffusionSolveResult:
    """Solve steady-state fluence field ``Phi`` via ``-div(D grad Phi) + μ_a Phi = S`` on the tet mesh.

    Assembles an H1 finite-element method (FEM) system with volumetric source
    density ``S_clean`` (per-tet, piecewise constant (L2, order-0) and either Robin
    boundary condition (extrapolated-flux leakage)
    (``robin_boundary_model="effective_refractive_boundary"`` with boundary term
    ``(D / L) u v`` on ``ds``) or Dirichlet ``Phi=0`` when
    ``use_dirichlet_zero_debug=True``.

    Requires a live ``diffusion_mesh.netgen_mesh`` handle from
    :func:`generate_diffusion_mesh`; cache-only meshes without Netgen cannot
    run the solve. ``S_clean`` must match ``diffusion_mesh.n_tets``.

    Parameters
    ----------
    diffusion_mesh:
        Coarse tet mesh with live Netgen handle.
    S_clean:
        Volumetric source density ``S(x)`` per tet, shape ``[n_tets]``.
    D:
        Diffusion coefficient (mesh units² / time scale implicit in ``S``).
    mu_a:
        Absorption coefficient ``mu_a`` (1 / mesh units).
    extrapolation_length:
        Robin boundary condition length ``L > 0`` (mesh units).
    robin_boundary_model:
        Must be :data:`ROBIN_BOUNDARY_MODEL` unless using Dirichlet debug mode.
    fem_order:
        H1 element order (default linear).
    use_dirichlet_zero_debug:
        Replace Robin boundary condition (BC) with homogeneous Dirichlet ``Phi=0`` for debugging.

    Returns
    -------
    DiffusionSolveResult

    Raises
    ------
    ImportError
        When NGSolve is unavailable (install ``fem`` extra).
    ValueError
        Shape mismatch, missing Netgen handle, unsupported boundary condition (BC), or ``L <= 0``.

    Notebook / protocol:
        M4C FEM diffusion; Robin boundary condition default, Dirichlet debug optional.

    See also:
        :func:`~gummybear.optics.diffuse_sampling.sample_phi_at_hit_points` — camera fluence sampling.
        :func:`~gummybear.optics.hybrid_compose.compose_hybrid_image` — combine direct + diffuse channels.
    """
    ngsolve = _require_ngsolve()
    from ngsolve import (
        H1,
        BilinearForm,
        LinearForm,
        GridFunction,
        L2,
        dx,
        ds,
        grad,
        Mesh as NGMesh,
    )

    if (
        robin_boundary_model != ROBIN_BOUNDARY_MODEL
        and not use_dirichlet_zero_debug
    ):
        raise ValueError(
            f"Unsupported robin_boundary_model={robin_boundary_model!r}; "
            f"expected {ROBIN_BOUNDARY_MODEL!r}"
        )

    S_clean = np.asarray(S_clean, dtype=float)
    if S_clean.shape != (diffusion_mesh.n_tets,):
        raise ValueError(
            f"S_clean must have shape [{diffusion_mesh.n_tets}], got {S_clean.shape}"
        )
    if diffusion_mesh.netgen_mesh is None:
        raise ValueError(
            "DiffusionMesh.netgen_mesh is required for NGSolve solve. "
            "Regenerate the mesh with generate_diffusion_mesh() "
            "(cache-only meshes without a Netgen handle cannot run M4C)."
        )

    ngmesh = NGMesh(diffusion_mesh.netgen_mesh)

    # Continuous H1 finite elements are used for the fluence field.
    # This is the standard approach for diffusion problems and proved robust
    # throughout development. Higher-order or alternative spaces were not
    # required for the objectives of this project.
    #
    # In practice, first-order (order=1) elements are typically used here.
    # The diffusion operator depends on spatial gradients of the fluence field,
    # making an order-0 representation unsuitable. 

    if use_dirichlet_zero_debug:
        V = H1(ngmesh, order=int(fem_order), dirichlet=".*")
        bc_model = "dirichlet_zero_debug"
    else:
        V = H1(ngmesh, order=int(fem_order))
        bc_model = robin_boundary_model

    u, v = V.TnT()
    a = BilinearForm(V, symmetric=True)
    a += float(D) * grad(u) * grad(v) * dx
    a += float(mu_a) * u * v * dx
    if not use_dirichlet_zero_debug:
        L = float(extrapolation_length)
        if L <= 0.0:
            raise ValueError("extrapolation_length must be positive")
        a += (float(D) / L) * u * v * ds
    a.Assemble()

    # The radiation source is a per-element quantity: ray-tetrahedron
    # intersections are calculated individually but their contributions
    # are accumulated at the tetrahedron level. An order-0 representation
    # therefore provides one source value per tetrahedron and matches the
    # native source format.
    S0 = GridFunction(L2(ngmesh, order=0)) 
    svec = S0.vec.FV().NumPy()
    if len(svec) != len(S_clean):
        raise RuntimeError(
            f"L2(order=0) dof count {len(svec)} != n_tets {len(S_clean)}"
        )
    svec[:] = S_clean

    f = LinearForm(V)
    f += S0 * v * dx
    f.Assemble()

    gfu = GridFunction(V)
    inv = a.mat.Inverse(freedofs=V.FreeDofs(), inverse="sparsecholesky")
    gfu.vec.data = inv * f.vec

    Phi_raw = np.asarray(gfu.vec.FV().NumPy(), dtype=float).copy()
    if len(Phi_raw) == diffusion_mesh.n_nodes:
        Phi_nodes = Phi_raw
    else:
        Phi_nodes = _reindex_phi_to_diffusion_nodes(
            ngmesh, diffusion_mesh.nodes, Phi_raw
        )

    Phi_tets = Phi_nodes[diffusion_mesh.tets].mean(axis=1)

    keys = operator_cache_keys(
        diffusion_mesh,
        D=D,
        mu_a=mu_a,
        extrapolation_length=extrapolation_length,
        robin_boundary_model=bc_model,
        fem_order=fem_order,
    )

    resid = a.mat.CreateColVector()
    resid.data = a.mat * gfu.vec - f.vec
    f_norm = float(np.linalg.norm(np.asarray(f.vec.FV().NumPy())))
    r_norm = float(np.linalg.norm(np.asarray(resid.FV().NumPy())))
    residual = (r_norm / f_norm) if f_norm > 0 else r_norm

    return DiffusionSolveResult(
        Phi_nodes=Phi_nodes,
        Phi_tets=Phi_tets,
        residual_norm=residual,
        operator_cache_key=keys,
        robin_boundary_model=bc_model,
        extrapolation_length=float(extrapolation_length),
        D=float(D),
        mu_a=float(mu_a),
        diagnostics={
            "ndof": int(V.ndof),
            "n_tets": int(diffusion_mesh.n_tets),
            "operator_key_hash": hashlib.sha256(
                json.dumps(keys, sort_keys=True).encode()
            ).hexdigest()[:16],
            "ngsolve_version": getattr(ngsolve, "__version__", None),
        },
    )


def sample_phi_at_points(
    diffusion_mesh: DiffusionMesh,
    Phi_nodes: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Sample nodal fluence field ``Phi`` at points via nearest-tet mean (legacy fallback).

    For each point, finds the closest tet centroid and returns the mean of
    the four corner nodal ``Phi_nodes`` values. Non-finite points return NaN.
    Prefer :func:`~gummybear.optics.diffuse_sampling.interpolate_phi_nodes_to_points`
    for barycentric interpolation.

    Parameters
    ----------
    diffusion_mesh:
        Coarse tet mesh.
    Phi_nodes:
        Nodal fluence, shape ``[n_nodes]``.
    points:
        Query locations, shape ``[N, 3]``.

    Returns
    -------
    np.ndarray, shape ``[N]``
        Sampled fluence; NaN where the point is non-finite.

    Notebook / protocol:
        M4D legacy ``interpolate=False`` path.
    """
    points = np.asarray(points, dtype=float)
    Phi_nodes = np.asarray(Phi_nodes, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N, 3]")
    if Phi_nodes.shape != (diffusion_mesh.n_nodes,):
        raise ValueError("Phi_nodes length must match diffusion_mesh.n_nodes")

    cents = diffusion_mesh.centroids
    tets = diffusion_mesh.tets
    out = np.full(len(points), np.nan, dtype=float)
    for i, p in enumerate(points):
        if not np.all(np.isfinite(p)):
            continue
        e = int(np.argmin(np.sum((cents - p) ** 2, axis=1)))
        out[i] = float(np.mean(Phi_nodes[tets[e]]))
    return out
