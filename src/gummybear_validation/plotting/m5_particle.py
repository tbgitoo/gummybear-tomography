"""Matplotlib helpers for Milestone 5 particle / source-delta / M5D notebooks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from .source_delta_3d import plot_active_element_scalar_3d, source_active_element_mask, source_delta_plot_panels


def plot_source_delta_triptych(
    diff_mesh,
    *,
    E_clean_elem: np.ndarray,
    source_delta,
    title: str,
    view_elev: float = 24.0,
    view_azim: float = -58.0,
    mesh_face_alpha: float = 0.13,
    mesh_edge_color: str = "0.60",
    mesh_linewidth: float = 0.20,
    point_size: float = 42.0,
    relative_floor: float = 1e-5,
    print_totals: bool = True,
) -> None:
    """Three-panel 3D view of background / particle-scatter / total source delta."""
    delta_bg = np.asarray(source_delta.delta_E_background_elem, dtype=float)
    delta_ps = np.asarray(source_delta.delta_E_particle_scat_elem, dtype=float)
    delta_tot = np.asarray(source_delta.delta_E_transport_elem, dtype=float)
    e_part = np.asarray(source_delta.E_particle_elem, dtype=float)

    active_mask = source_active_element_mask(
        E_clean_elem,
        e_part,
        delta_bg,
        delta_ps,
        delta_tot,
    )
    panels = source_delta_plot_panels(delta_bg, delta_ps, delta_tot)

    fig = plt.figure(figsize=(17, 5.5))
    mappables: list[tuple[Any, Any, str]] = []
    for col, panel in enumerate(panels):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        sc = plot_active_element_scalar_3d(
            ax,
            diff_mesh,
            panel["values"],
            panel["title"],
            active_mask=active_mask,
            signed=panel["signed"],
            cmap=panel["cmap"],
            mesh_face_alpha=mesh_face_alpha,
            mesh_edge_color=mesh_edge_color,
            mesh_linewidth=mesh_linewidth,
            point_size=point_size,
            relative_floor=relative_floor,
            view_elev=view_elev,
            view_azim=view_azim,
        )
        mappables.append((ax, sc, panel["colorbar_label"]))

    for ax, sc, label in mappables:
        if sc is not None:
            fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.06, label=label)

    fig.suptitle(title, fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    if print_totals:
        print("Source-delta totals:")
        print("  delta_E_background_elem    =", float(delta_bg.sum()))
        print("  delta_E_particle_scat_elem =", float(delta_ps.sum()))
        print("  delta_E_transport_elem     =", float(delta_tot.sum()))


def plot_m5d_propagation(
    result,
    title: str,
    *,
    view_elev: float = 10.0,
    view_azim: float = -90.0,
) -> None:
    """Source-density delta → fluence delta → diffuse camera delta for one M5D result."""
    diff_mesh = result.scene.diff_mesh
    camera_mask = result.scene.camera_mask
    S_delta = result.source.S_delta
    active_source = result.source.active_source_mask
    Phi_delta_tets = result.diffusion.Phi_delta_tets
    camera_delta = result.camera.Delta_I_diffuse

    assert np.all(camera_delta[~camera_mask] == 0.0)

    fig = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    ax_source = fig.add_subplot(1, 3, 1, projection="3d")
    ax_phi = fig.add_subplot(1, 3, 2, projection="3d")
    ax_camera = fig.add_subplot(1, 3, 3)

    m_source = plot_active_element_scalar_3d(
        ax_source,
        diff_mesh,
        S_delta,
        "M5C source-density delta",
        active_mask=active_source,
        signed=True,
        cmap="coolwarm",
        relative_floor=1e-5,
        view_elev=view_elev,
        view_azim=view_azim,
        mesh_face_alpha=0.13,
        mesh_edge_color="0.60",
        mesh_linewidth=0.20,
        point_size=42,
    )
    m_phi = plot_active_element_scalar_3d(
        ax_phi,
        diff_mesh,
        Phi_delta_tets,
        "Diffused fluence delta",
        signed=True,
        cmap="coolwarm",
        relative_floor=1e-4,
        view_elev=view_elev,
        view_azim=view_azim,
        mesh_face_alpha=0.13,
        mesh_edge_color="0.60",
        mesh_linewidth=0.20,
        point_size=42,
    )

    camera_values = np.asarray(camera_delta[camera_mask], dtype=float)
    nonzero = np.abs(camera_values[np.abs(camera_values) > 0.0])
    if nonzero.size == 0:
        camera_limit = np.finfo(float).eps
        camera_linthresh = np.finfo(float).eps
    else:
        camera_limit = max(float(np.percentile(nonzero, 99.5)), np.finfo(float).eps)
        camera_linthresh = max(
            float(np.percentile(nonzero, 35)),
            camera_limit * 1e-3,
            np.finfo(float).eps,
        )

    camera_norm = SymLogNorm(
        linthresh=camera_linthresh,
        vmin=-camera_limit,
        vmax=camera_limit,
        base=10,
    )
    camera_display = np.ma.array(camera_delta, mask=~camera_mask)
    m_camera = ax_camera.imshow(
        camera_display,
        cmap="coolwarm",
        norm=camera_norm,
        origin="lower",
    )
    ax_camera.set_title("Camera diffuse delta\nparticle - clean")
    ax_camera.axis("off")

    if m_source is not None:
        fig.colorbar(m_source, ax=ax_source, shrink=0.68, pad=0.04, label="delta S")
    if m_phi is not None:
        fig.colorbar(m_phi, ax=ax_phi, shrink=0.68, pad=0.04, label="delta Phi")
    fig.colorbar(m_camera, ax=ax_camera, shrink=0.78, pad=0.04, label="delta I diffuse")
    fig.suptitle(title, fontsize=15)
    plt.show()


def plot_m5d_final_images(
    result,
    title: str,
    particle_panel_title: str,
) -> None:
    """Side-by-side clean vs particle camera intensity (shared scale from zero)."""
    camera_mask = result.scene.camera_mask

    if result.camera.clean_hybrid is not None and result.camera.particle_hybrid is not None:
        I_clean = np.asarray(result.camera.clean_hybrid.I_total, dtype=float)
        I_particle = np.asarray(result.camera.particle_hybrid.I_total, dtype=float)
        image_label = "final hybrid camera intensity"
    else:
        I_clean = np.asarray(result.camera.clean_diffuse.I_diffuse, dtype=float)
        I_particle = np.asarray(result.camera.particle_diffuse.I_diffuse, dtype=float)
        image_label = "diffuse camera intensity"

    assert np.all(I_clean[~camera_mask] == 0.0)
    assert np.all(I_particle[~camera_mask] == 0.0)

    joint = np.concatenate([I_clean[camera_mask], I_particle[camera_mask]])
    display_vmin = 0.0
    display_vmax = max(float(np.percentile(joint, 99)), np.finfo(float).eps)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    m0 = axes[0].imshow(
        I_clean,
        cmap="magma",
        vmin=display_vmin,
        vmax=display_vmax,
        origin="lower",
    )
    axes[1].imshow(
        I_particle,
        cmap="magma",
        vmin=display_vmin,
        vmax=display_vmax,
        origin="lower",
    )
    axes[0].set_title("Clean")
    axes[1].set_title(particle_panel_title)
    for ax in axes:
        ax.axis("off")

    fig.colorbar(m0, ax=axes, shrink=0.78, pad=0.03, label=image_label)
    fig.suptitle(title, fontsize=15)
    plt.show()

    print("Final camera intensity comparison:")
    print("  image type           =", image_label)
    print("  clean sum            =", float(I_clean[camera_mask].sum()))
    print("  particle sum         =", float(I_particle[camera_mask].sum()))
    print(
        "  particle - clean sum =",
        float((I_particle - I_clean)[camera_mask].sum()),
    )
