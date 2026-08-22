"""Matplotlib helpers for Milestone 4 diffusion validation notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gummybear.optics.hybrid_compose import HybridImageResult


def _display_limits(img: np.ndarray, mask: np.ndarray, q: tuple[float, float] = (2, 98)) -> tuple[float, float]:
    vals = np.asarray(img)[mask]
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        return 0.0, 1.0
    lo, hi = np.percentile(vals, q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def plot_diffusion_centroids_3d(
    centroids: np.ndarray,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 8.0),
    point_size: float = 2.0,
) -> None:
    """Scatter tet centroids in 3D."""
    c = np.asarray(centroids, dtype=float)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=point_size)
    ax.set_title(title or f"{len(c)} tetrahedra")
    fig.tight_layout()
    plt.show()


def plot_surface_with_diffusion_centroids(
    surface_mesh,
    diff_mesh,
    *,
    figsize: tuple[float, float] = (10.0, 10.0),
    surface_alpha: float = 0.2,
) -> None:
    """Overlay STL surface (transparent) and diffusion tet centroids."""
    surface = surface_mesh
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        surface.vertices[:, 0],
        surface.vertices[:, 1],
        surface.vertices[:, 2],
        triangles=surface.faces,
        alpha=surface_alpha,
        color="lightgray",
    )
    c = np.asarray(diff_mesh.centroids, dtype=float)
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=2, c="red")
    ax.set_title("STL surface + diffusion tet centroids")
    fig.tight_layout()
    plt.show()


def plot_phi_on_nodes(
    nodes: np.ndarray,
    phi: np.ndarray,
    *,
    log_scale: bool = True,
    title: str = "Phi on diffusion nodes",
    figsize: tuple[float, float] = (8.0, 8.0),
    point_size: float = 6.0,
) -> None:
    """3D scatter of nodal fluence (optional log10)."""
    nodes = np.asarray(nodes, dtype=float)
    phi = np.asarray(phi, dtype=float)
    values = np.log10(phi + 1e-12) if log_scale else phi
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=values, s=point_size, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="log10(Phi)" if log_scale else "Phi")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_profile_along_ray(
    centroids: np.ndarray,
    values: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    ylabel: str,
    title: str,
) -> None:
    """Line plot of a per-tet scalar ordered along a ray direction."""
    centroids = np.asarray(centroids, dtype=float)
    values = np.asarray(values, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    direction = direction / np.linalg.norm(direction)
    active = values > 0
    if not np.any(active):
        print("no active values to plot")
        return
    t = (centroids - p0) @ direction
    active_idx = np.where(active)[0]
    order = active_idx[np.argsort(t[active_idx])]
    plt.figure()
    plt.plot(t[order], values[order], marker="o")
    plt.xlabel("position along ray")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_deposition_scene(
    centroids: np.ndarray,
    e_scat_elem: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    figsize: tuple[float, float] = (8.0, 8.0),
) -> None:
    """3D view: inactive tets grey, active tets colored by E_scat_elem, ray chord."""
    centroids = np.asarray(centroids, dtype=float)
    e_scat = np.asarray(e_scat_elem, dtype=float)
    active = e_scat > 0
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="lightgrey", s=5, alpha=0.35)
    if np.any(active):
        sc = ax.scatter(
            centroids[active, 0],
            centroids[active, 1],
            centroids[active, 2],
            c=e_scat[active],
            cmap="hot",
            s=20,
        )
        plt.colorbar(sc, ax=ax, label="E_scat_elem")
    ax.scatter([p0[0]], [p0[1]], [p0[2]], c="green", s=40, label="start")
    ax.scatter([p1[0]], [p1[1]], [p1[2]], c="blue", s=40, label="end")
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], "b-", linewidth=2)
    ax.set_title("Deposition along synthetic axis ray")
    fig.tight_layout()
    plt.show()


def plot_camera_scalar(img: np.ndarray, *, title: str, figsize: tuple[float, float] = (6.0, 6.0)) -> None:
    """Imshow a camera-grid scalar with colorbar."""
    plt.figure(figsize=figsize)
    im = plt.imshow(np.asarray(img), origin="lower")
    plt.colorbar(im, label=title)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_hybrid_panels(
    hybrid: HybridImageResult,
    camera_mask: np.ndarray,
    *,
    alpha_label: float | None = None,
    forward_model_tier: str = "",
    show_alpha_sweep: bool = False,
    alpha_sweep: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    figsize_row: tuple[float, float] = (14.0, 4.5),
) -> None:
    """Three-panel I_direct / I_diffuse / I_total and optional alpha sweep."""
    mask = np.asarray(camera_mask, dtype=bool)
    alpha_tag = alpha_label if alpha_label is not None else hybrid.alpha
    panels = [
        ("I_direct", hybrid.I_direct),
        ("I_diffuse", hybrid.I_diffuse),
        (f"I_total (alpha={alpha_tag:g})", hybrid.I_total),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize_row)
    for ax, (title, img) in zip(axes, panels, strict=True):
        vmin, vmax = _display_limits(img, mask)
        im = ax.imshow(img, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    suptitle = "M4E hybrid compose"
    if forward_model_tier:
        suptitle += f" — {forward_model_tier}"
    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()

    if not show_alpha_sweep:
        return

    from gummybear.optics.hybrid_compose import compose_hybrid_image

    fig, axes = plt.subplots(1, len(alpha_sweep), figsize=(14, 3.5))
    for ax, a in zip(axes, alpha_sweep, strict=True):
        h = compose_hybrid_image(hybrid.I_direct, hybrid.I_diffuse, alpha=a, camera_mask=mask)
        vmin, vmax = _display_limits(h.I_total, mask)
        im = ax.imshow(h.I_total, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"alpha={a:g}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Alpha sweep (alpha=0 recovers I_diffuse)")
    fig.tight_layout()
    plt.show()
