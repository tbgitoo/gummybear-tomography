"""Milestone 4 diffusion mesh, deposition, and M4E notebook helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gummybear.optics.diffusion_mesh import DiffusionMesh
from gummybear.optics.source_deposition import (
    RaySegmentBundle,
    SourceDepositionResult,
    make_synthetic_axis_ray,
)
from gummybear.paths import display_path, display_text_paths
from gummybear.rays.source import SourceRayBundle


def print_diffusion_mesh_summary(diff_mesh: DiffusionMesh, *, label: str = "diffusion mesh") -> None:
    """Print tet/node counts, bounds, and cache metadata."""
    print(f"{label}: n_tets={diff_mesh.n_tets}  n_nodes={diff_mesh.n_nodes}")
    print(f"  bounds min={diff_mesh.bounds[0]}  max={diff_mesh.bounds[1]}")
    meta = diff_mesh.metadata
    print(f"  meshing_method={meta.meshing_method}  netgen_mesh attached={diff_mesh.netgen_mesh is not None}")


def assert_live_netgen_mesh(diff_mesh: DiffusionMesh) -> None:
    """Require a live Netgen handle for NGSolve (npz-only cache reload is insufficient)."""
    if diff_mesh.netgen_mesh is None:
        raise AssertionError(
            "DiffusionMesh.netgen_mesh is missing; regenerate without cache_dir "
            "(npz-only cache cannot run solve_diffusion)."
        )


def make_centroid_axis_ray(
    diff_mesh: DiffusionMesh,
    *,
    axis: str = "x",
    intensity: float = 1.0,
    margin_fraction: float = 0.1,
) -> tuple[RaySegmentBundle, np.ndarray, np.ndarray]:
    """Build a synthetic axis-aligned ray spanning the diffusion mesh centroid bbox."""
    centroids = np.asarray(diff_mesh.centroids, dtype=float)
    bounds_min = centroids.min(axis=0)
    bounds_max = centroids.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    margin = margin_fraction * float(np.linalg.norm(bounds_max - bounds_min))
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    p0 = center.copy()
    p1 = center.copy()
    p0[axis_idx] = bounds_min[axis_idx] - margin
    p1[axis_idx] = bounds_max[axis_idx] + margin
    ray = make_synthetic_axis_ray(start=tuple(p0), end=tuple(p1), intensity=intensity)
    return ray, p0, p1


def print_deposition_summary(result: SourceDepositionResult) -> None:
    """Print M4B energy bookkeeping and source-density stats."""
    print(f"total_ballistic_input: {result.total_ballistic_input:.6f}")
    print(f"total_scattered: {result.total_scattered:.6f}")
    print(f"total_absorbed: {result.total_absorbed:.6f}")
    print(f"remaining_direct_energy: {result.remaining_direct_energy:.6f}")
    print(
        "S_clean min/mean/max:",
        float(result.S_clean.min()),
        float(result.S_clean.mean()),
        float(result.S_clean.max()),
    )
    print("E_scat nonzero:", int(np.count_nonzero(result.E_scat_elem)))
    print("S_clean nonzero:", int(np.count_nonzero(result.S_clean)))


def assert_deposition_conservation(
    result: SourceDepositionResult,
    volumes: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> None:
    """Assert sum(S_clean * volume) matches reported scattered energy."""
    volumes = np.asarray(volumes, dtype=float)
    reconstructed = float(np.sum(result.S_clean * volumes))
    print("reconstructed_scattered:", reconstructed)
    print("reported total_scattered:", result.total_scattered)
    np.testing.assert_allclose(reconstructed, result.total_scattered, rtol=rtol, atol=atol)


def assert_deposition_sanity(result: SourceDepositionResult, n_tets: int) -> None:
    """Shape, finiteness, and non-negativity checks for deposition output."""
    assert result.S_clean.shape == (n_tets,)
    assert result.E_scat_elem.shape == (n_tets,)
    assert np.all(np.isfinite(result.S_clean))
    assert np.all(np.isfinite(result.E_scat_elem))
    assert np.all(result.S_clean >= 0)
    assert np.all(result.E_scat_elem >= 0)
    assert result.total_scattered >= 0
    assert result.total_absorbed >= 0
    assert result.remaining_direct_energy >= 0


def build_diffusion_ray_subset(
    source_rays: SourceRayBundle,
    n_diffuse: int,
    *,
    seed: int = 0,
) -> tuple[SourceRayBundle, float]:
    """Random subset for expensive M4B/M4C with weight scaling preserved in expectation."""
    n_diffuse = min(int(n_diffuse), source_rays.n_rays)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(source_rays.n_rays, size=n_diffuse, replace=False))
    weight_scale = float(source_rays.n_rays) / float(n_diffuse)
    subset = SourceRayBundle(
        origins=source_rays.origins[idx],
        directions=source_rays.directions[idx],
        weights=source_rays.weights[idx] * weight_scale,
        sample_shape=None,
        metadata={
            "role": "diffusion_subset",
            "parent_n_rays": int(source_rays.n_rays),
            "subset_n_rays": int(n_diffuse),
            "weight_scale": weight_scale,
            "seed": int(seed),
        },
    )
    return subset, weight_scale


def write_m4e_artifacts(
    out_dir: Path,
    *,
    hybrid,
    deposit: SourceDepositionResult,
    solved,
    metadata: dict[str, Any],
    camera_mask: np.ndarray | None = None,
) -> None:
    """Persist M4E npz images and JSON metadata under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "I_direct": hybrid.I_direct,
        "I_diffuse": hybrid.I_diffuse,
        "I_total": hybrid.I_total,
        "S_clean": deposit.S_clean,
        "Phi_nodes": solved.Phi_nodes,
    }
    if camera_mask is not None:
        payload["camera_mask"] = np.asarray(camera_mask, dtype=bool)
    np.savez_compressed(out_dir / "m4e_images.npz", **payload)
    meta_path = out_dir / "m4e_run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print("wrote", display_path(out_dir / "m4e_images.npz"))
    print("wrote", display_path(meta_path))


def print_m4e_metadata(metadata: dict[str, Any]) -> None:
    """Print run metadata with paths rewritten for notebook-safe display."""
    print(display_text_paths(json.dumps(metadata, indent=2)))


def assert_m4e_hybrid_checks(
    hybrid,
    I_diffuse: np.ndarray,
    camera_mask: np.ndarray,
    *,
    forward_model: str = "m4_refractive_diffusion",
) -> None:
    """Notebook success checks for M4E hybrid composition."""
    mask = np.asarray(camera_mask, dtype=bool)
    h, w = hybrid.I_direct.shape
    assert hybrid.I_direct.shape == hybrid.I_diffuse.shape == hybrid.I_total.shape == (h, w)
    assert np.all(np.isfinite(hybrid.I_total))
    assert np.all(hybrid.I_direct[~mask] == 0.0)
    assert np.all(hybrid.I_diffuse[~mask] == 0.0)
    assert np.all(hybrid.I_total[~mask] == 0.0)

    from gummybear.optics.hybrid_compose import compose_hybrid_image

    zero = compose_hybrid_image(hybrid.I_direct, I_diffuse, alpha=0.0, camera_mask=mask)
    np.testing.assert_allclose(zero.I_total, I_diffuse)

    if float(I_diffuse[mask].sum()) > 0:
        assert float(hybrid.I_total[mask].mean()) >= float(hybrid.I_direct[mask].mean()) - 1e-12

    assert hybrid.forward_model == forward_model
    assert hybrid.metadata["alpha"] == float(hybrid.alpha)
    print("M4E checks passed.")


def m4e_metadata_template(
    *,
    run_id: str,
    forward_model: str,
    alpha: float,
    exitance_scale: float,
    direct_scale: float,
    stl_path: Path,
    target_elements: int,
    diff_mesh: DiffusionMesh,
    material,
    extrapolation_length: float,
    robin_boundary_model: str,
    light,
    source_rays: SourceRayBundle,
    diffusion_rays: SourceRayBundle,
    weight_scale: float,
    ray_seed: int,
    camera,
    sample_shape: tuple[int, int],
    hit_fraction: float,
    deposit: SourceDepositionResult,
    n_segments: int,
    hybrid,
) -> dict[str, Any]:
    """Build JSON-serializable M4E run metadata."""
    h, w = sample_shape
    return {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forward_model": forward_model,
        "composition": "I_total = alpha * I_direct + I_diffuse",
        "alpha": float(alpha),
        "exitance_scale": float(exitance_scale),
        "direct_scale": float(direct_scale),
        "stl_path": display_path(stl_path),
        "target_elements": int(target_elements),
        "n_tets": int(diff_mesh.n_tets),
        "n_nodes": int(diff_mesh.n_nodes),
        "material": {
            "n_refractive": float(material.n_refractive),
            "mu_s": float(material.mu_s),
            "mu_a": float(material.mu_a),
            "g": float(material.g),
            "D": float(material.diffusion_coefficient),
        },
        "extrapolation_length": float(extrapolation_length),
        "robin_boundary_model": robin_boundary_model,
        "light": {
            "type": "point",
            "position": list(map(float, light.position)),
            "intensity": float(light.intensity),
            "n_direct_rays": int(source_rays.n_rays),
            "n_diffuse_rays": int(diffusion_rays.n_rays),
            "diffuse_subset_weight_scale": float(weight_scale),
            "ray_seed": int(ray_seed),
        },
        "camera": {
            "type": "pinhole",
            "position": list(map(float, camera.camera_position)),
            "look_at": list(map(float, camera.look_at)),
            "fov_deg": float(camera.fov_deg),
            "resolution": [int(h), int(w)],
            "hit_fraction": float(hit_fraction),
        },
        "energy": {
            "total_ballistic_input": float(deposit.total_ballistic_input),
            "total_scattered": float(deposit.total_scattered),
            "total_absorbed": float(deposit.total_absorbed),
            "remaining_direct_energy": float(deposit.remaining_direct_energy),
            "n_segments": int(n_segments),
        },
        "components": dict(hybrid.metadata.get("components", {})),
    }
