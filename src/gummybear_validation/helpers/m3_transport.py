"""Milestone 3 source-ray and face-transport diagnostics for validation notebooks."""

from __future__ import annotations

import numpy as np

from gummybear.optics.face_transport import FaceOpticalState
from gummybear.rays.source import SourceRayBundle


def assert_source_ray_bundle_invariants(source_rays: SourceRayBundle) -> None:
    """Assert ``SourceRayBundle`` shape, finiteness, unit directions, and weights."""
    dir_norms = np.linalg.norm(source_rays.directions, axis=1)
    assert source_rays.origins.shape[1] == 3
    assert source_rays.directions.shape == source_rays.origins.shape
    assert source_rays.weights.shape == (source_rays.n_rays,)
    assert np.all(np.isfinite(source_rays.origins))
    assert np.all(np.isfinite(source_rays.directions))
    assert np.all(np.isfinite(source_rays.weights))
    assert np.allclose(dir_norms, 1.0)
    assert np.all(source_rays.weights >= 0)


def print_source_hit_summary(
    source_rays: SourceRayBundle,
    source_valid: np.ndarray,
    source_depth: np.ndarray,
    source_entry_faces: np.ndarray,
) -> None:
    """Print first-surface hit counts and entry-face index range."""
    n_hit = int(np.count_nonzero(source_valid))
    print(f"rays: {source_rays.n_rays}")
    print(f"hits: {n_hit}")
    print(f"hit fraction: {n_hit / max(1, source_rays.n_rays):.4f}")
    if n_hit:
        hit_faces = source_entry_faces[source_valid]
        print(f"entry face min/max: {hit_faces.min()} {hit_faces.max()}")


def assert_source_hit_contract(
    source_rays: SourceRayBundle,
    source_valid: np.ndarray,
    source_depth: np.ndarray,
    source_entry_faces: np.ndarray,
) -> None:
    """Assert ``first_visible_hits`` outputs for a source bundle."""
    assert source_valid.shape == (source_rays.n_rays,)
    assert source_depth.shape == (source_rays.n_rays,)
    assert source_entry_faces.shape == (source_rays.n_rays,)
    assert np.all(source_entry_faces[~source_valid] < 0)
    assert np.all(source_entry_faces[source_valid] >= 0)
    assert np.all(np.isfinite(source_depth[source_valid]))
    assert np.count_nonzero(source_valid) > 0


def print_entry_refraction_summary(
    source_rays: SourceRayBundle,
    source_valid: np.ndarray,
    source_entry_faces: np.ndarray,
    internal_dirs: np.ndarray,
    internal_valid: np.ndarray,
    mesh,
) -> None:
    """Print Snell entry-refraction fractions, bend angles, and inward check."""
    n_source_hits = int(np.count_nonzero(source_valid & (source_entry_faces >= 0)))
    n_internal = int(np.count_nonzero(internal_valid))
    print(f"source entry hits: {n_source_hits}")
    print(f"internal valid: {n_internal}")
    if n_source_hits:
        print(f"fraction refracted: {n_internal / n_source_hits:.4f}")

    if n_internal == 0:
        return

    norms = np.linalg.norm(internal_dirs[internal_valid], axis=1)
    print(f"internal direction norm min/max/mean: {norms.min():.6f} {norms.max():.6f} {norms.mean():.6f}")

    incoming = source_rays.directions[internal_valid]
    refracted = internal_dirs[internal_valid]
    dot = np.clip(np.sum(incoming * refracted, axis=1), -1.0, 1.0)
    bend_deg = np.degrees(np.arccos(dot))
    print(f"bend angle deg min/mean/max: {bend_deg.min():.2f} {bend_deg.mean():.2f} {bend_deg.max():.2f}")

    entry_normals = np.asarray(mesh.face_normals)[source_entry_faces[internal_valid]]
    inward_dot = np.sum(refracted * entry_normals, axis=1)
    print(f"dot(internal, entry_normal) mean: {inward_dot.mean():.4f}")
    print(f"fraction pointing inward: {np.mean(inward_dot < 0.0):.4f}")


def print_face_state_summary(state: FaceOpticalState, *, label: str = "face state") -> None:
    """Print coverage and energy summaries for a ``FaceOpticalState``."""
    n_valid = int(np.count_nonzero(state.valid))
    print(f"{label}: valid faces={n_valid}")
    print(f"  total energy={state.face_energy.sum():.6f}  max energy={state.face_energy.max():.6f}")
    print(f"  max hit_count={state.hit_count.max()}")
