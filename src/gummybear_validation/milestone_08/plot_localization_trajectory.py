"""3D localisation-trajectory plots inside the phantom mesh."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from trimesh import Trimesh

from gummybear.particles.plotting_helpers import plot_sphere, set_axes_equal


def plot_localization_trajectory_in_mesh(
    ax: Axes3D,
    surface_mesh: Trimesh,
    particle: Any,
    trajectory: np.ndarray | Sequence[Sequence[float]],
    *,
    target: np.ndarray | Sequence[float] | None = None,
    mesh_facecolor: str = "lightgray",
    mesh_edgecolor: str = "tab:gray",
    mesh_linewidth: float = 0.08,
    mesh_alpha: float = 0.10,
    particle_color: str = "darkgrey",
    particle_alpha: float = 0.35,
    trajectory_color: str = "tab:red",
    trajectory_linewidth: float = 3.5,
    start_color: str = "tab:blue",
    end_color: str = "tab:orange",
    target_color: str = "tab:green",
    point_size: float = 70.0,
    title: Optional[str] = None,
    show_legend: bool = True,
    equal_aspect: bool = True,
) -> None:
    """Plot a predicted particle trajectory inside a translucent phantom mesh.

    Draws the surface hull, true analytic sphere, trajectory polyline, start/end
    markers, and optional target star. Visualization only—does not affect training
    or forward-model state.

    Args:
        ax: Matplotlib 3D axis (``projection="3d"``).
        surface_mesh: Phantom ``trimesh.Trimesh`` for the translucent hull.
        particle: Analytic sphere as ``ParticleSphere`` or ``{"center", "radius"}``.
        trajectory: Predicted centres with shape ``(T, 3)`` in world coordinates.
        target: Optional true centre ``(3,)`` marked with a star.
        mesh_facecolor, mesh_edgecolor, mesh_linewidth, mesh_alpha: Hull styling.
        particle_color, particle_alpha: True-sphere styling.
        trajectory_color, trajectory_linewidth: Predicted path styling.
        start_color, end_color, target_color: Marker colors.
        point_size: Scatter marker area for start/end/target.
        title: Optional axis title.
        show_legend: If True, draw a compact legend.
        equal_aspect: If True, call ``set_axes_equal`` after drawing.

    Raises:
        ValueError: When ``trajectory`` is empty or not shape ``(T, 3)``.
    """
    points = np.asarray(trajectory, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"trajectory must have shape (T, 3), got {points.shape}"
        )
    if points.shape[0] < 1:
        raise ValueError("trajectory must contain at least one point")

    triangles = np.asarray(surface_mesh.triangles, dtype=float)
    ax.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolor=mesh_facecolor,
            edgecolor=mesh_edgecolor,
            linewidth=mesh_linewidth,
            alpha=mesh_alpha,
        )
    )

    plot_sphere(
        ax,
        particle,
        color=particle_color,
        alpha=particle_alpha,
        filled=True,
        label="true particle",
    )

    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=trajectory_color,
        linewidth=trajectory_linewidth,
        solid_capstyle="round",
        label="prediction trajectory",
    )
    ax.scatter(
        points[0, 0],
        points[0, 1],
        points[0, 2],
        color=start_color,
        s=point_size,
        depthshade=False,
        label="start",
        zorder=5,
    )
    ax.scatter(
        points[-1, 0],
        points[-1, 1],
        points[-1, 2],
        color=end_color,
        s=point_size,
        depthshade=False,
        label="final prediction",
        zorder=5,
    )

    if target is not None:
        target_xyz = np.asarray(target, dtype=float).reshape(3)
        ax.scatter(
            target_xyz[0],
            target_xyz[1],
            target_xyz[2],
            color=target_color,
            s=point_size * 1.4,
            marker="*",
            depthshade=False,
            label="target centre",
            zorder=6,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title is not None:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", fontsize=9)
    if equal_aspect:
        set_axes_equal(ax)


def resolve_job_stl_path(
    job: Any,
    *,
    stl_root: str | None = None,
) -> str:
    """Resolve ``job.stl_path`` to an on-disk STL triangle mesh file.

    Tries the path as given, then ``stl_root / job.stl_path`` when ``stl_root``
    is provided.

    Args:
        job: Workbook job object with ``stl_path`` and optional ``sequence_id``.
        stl_root: Optional directory prefix for workbook-relative paths.

    Returns:
        str: Absolute or relative path to an existing STL triangle mesh file.

    Raises:
        FileNotFoundError: When neither candidate path exists.
    """
    from pathlib import Path

    raw = Path(str(job.stl_path))
    if raw.is_file():
        return str(raw)
    if stl_root is not None:
        candidate = Path(stl_root) / raw
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"STL not found for job {getattr(job, 'sequence_id', '?')!r}: "
        f"{job.stl_path!r} (stl_root={stl_root!r})"
    )


def particle_dict_from_catalog_row(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Build a ``plot_sphere``-compatible particle dict from a catalog row.

    Reads ``particle_x/y/z`` and ``particle_radius`` attributes (single-particle
    scalar columns on :class:`CatalogRow`).

    Args:
        row: Catalog row or namespace with particle label fields.

    Returns:
        dict: ``{"center": (x, y, z), "radius": float}``.

    Raises:
        ValueError: When ``particle_radius`` is missing.
    """
    center = (
        float(getattr(row, "particle_x")),
        float(getattr(row, "particle_y")),
        float(getattr(row, "particle_z")),
    )
    radius = getattr(row, "particle_radius", None)
    if radius is None:
        raise ValueError("catalog row is missing particle_radius")
    return {"center": center, "radius": float(radius)}
