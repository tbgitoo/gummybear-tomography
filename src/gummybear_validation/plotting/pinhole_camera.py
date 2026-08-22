"""2D matplotlib helpers for pinhole camera validation plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes

from gummybear.rays.camera import PinholeCameraConfig

from ..helpers.camera_helpers import pinhole_camera_basis

if TYPE_CHECKING:
    import trimesh


def plot_pinhole_wireframe(
    ax: Axes,
    mesh: trimesh.Trimesh,
    cam: PinholeCameraConfig,
    *,
    color: str = "C0",
    linewidth: float = 0.35,
    alpha: float = 0.85,
) -> None:
    """Project mesh edges onto the pinhole image plane.

    Uses the same basis and FOV as ``make_pinhole_rays``, so the wireframe aligns
    with ``hits_to_image`` rasters when both use ``origin=\"lower\"``.

    Args:
        ax: Matplotlib 2D axis (pixel coordinates).
        mesh: Surface mesh to draw.
        cam: Pinhole config defining pose and square FOV.
        color: Edge color passed to ``Axes.plot``.
        linewidth: Edge line width.
        alpha: Edge opacity.
    """
    eye, forward, right, up = pinhole_camera_basis(cam)
    res = cam.resolution
    half = np.tan(np.deg2rad(cam.fov_deg) / 2.0)
    verts = mesh.vertices

    for i0, i1 in mesh.edges:
        seg = verts[[i0, i1]]
        rel = seg - eye
        depth = rel @ forward
        if np.any(depth <= 1e-3):
            continue
        cols = ((rel @ right) / depth / half * 0.5 + 0.5) * (res - 1)
        rows = ((rel @ up) / depth / half * 0.5 + 0.5) * (res - 1)
        ax.plot(cols, rows, color=color, linewidth=linewidth, alpha=alpha)

    ax.set_xlim(0, res - 1)
    ax.set_ylim(0, res - 1)
    ax.set_aspect("equal")
    ax.axis("off")
