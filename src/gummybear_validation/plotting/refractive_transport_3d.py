"""3D matplotlib helpers for point-light refraction / ballistic transport scenes.

Draws the translucent STL surface hull, a point light, exterior source-to-entry
segments, in-object refracted chords, and optionally an analytic particle with
particle entry/exit markers. Visualization only — does not alter transport,
deposition, or diffusion state.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from trimesh import Trimesh

from gummybear.particles.plotting_helpers import plot_sphere

from .intersection_geometry import plot_segment, set_axes_equal


def add_transparent_surface_mesh(
    ax: Axes3D,
    surface_mesh: Trimesh,
    *,
    facecolor: str = "lightgray",
    edgecolor: str = "tab:gray",
    linewidth: float = 0.08,
    alpha: float = 0.10,
) -> Poly3DCollection:
    """Add a translucent, wire-edged STL surface mesh to a 3D axis.

    Matches the visual style used by particle / localisation mesh overlays
    (transparent faces + fine triangle edges).
    """
    triangles = np.asarray(surface_mesh.triangles, dtype=float)
    collection = Poly3DCollection(
        triangles,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(collection)
    return collection


def plot_point_light(
    ax: Axes3D,
    position: Sequence[float] | np.ndarray,
    *,
    color: str = "gold",
    size: float = 120.0,
    label: str | None = "point light",
) -> None:
    """Mark a point-light location."""
    p = np.asarray(position, dtype=float).reshape(3)
    ax.scatter(
        [p[0]],
        [p[1]],
        [p[2]],
        color=color,
        s=size,
        depthshade=False,
        label=label,
        zorder=6,
    )


def _as_xyz(obj: Any, names: Sequence[str], label: str) -> np.ndarray:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return np.asarray(obj[name], dtype=float)
        if hasattr(obj, name):
            return np.asarray(getattr(obj, name), dtype=float)
    raise AttributeError(
        f"Could not find {label}. Tried {list(names)} on {type(obj)!r}."
    )


def _event_field(event: Any, names: Sequence[str], label: str) -> Any:
    for name in names:
        if isinstance(event, dict) and name in event:
            return event[name]
        if hasattr(event, name):
            return getattr(event, name)
    raise AttributeError(
        f"Could not find {label}. Tried {list(names)} on {type(event)!r}."
    )


def plot_refractive_illumination_scene(
    ax: Axes3D,
    surface_mesh: Trimesh,
    light: Any,
    source_rays: Any,
    refract_result: Any,
    segments: Any,
    *,
    particle: Any | None = None,
    particle_events: Sequence[Any] | None = None,
    max_rays_to_draw: int = 64,
    max_particle_events_to_draw: int | None = 40,
    mesh_facecolor: str = "lightgray",
    mesh_edgecolor: str = "tab:gray",
    mesh_linewidth: float = 0.08,
    mesh_alpha: float = 0.10,
    light_color: str = "gold",
    light_size: float = 120.0,
    exterior_color: str = "tab:orange",
    exterior_linewidth: float = 0.7,
    interior_color: str = "tab:blue",
    interior_linewidth: float = 1.1,
    particle_color: str = "crimson",
    particle_alpha: float = 0.55,
    particle_entry_color: str = "limegreen",
    particle_exit_color: str = "deepskyblue",
    particle_point_size: float = 45.0,
    view_elev: float = 10.0,
    view_azim: float = -90.0,
    title: Optional[str] = None,
    show_legend: bool = True,
    equal_aspect: bool = True,
) -> None:
    """Plot phantom mesh with point light, refracted rays, and optional particle.

    Draws:

    - translucent wire-edged surface mesh,
    - point-light marker,
    - exterior ballistic segments (source origin → mesh entry),
    - in-object refracted segments (mesh entry → mesh exit),
    - optional colored analytic particle sphere,
    - optional particle entry / exit markers (mesh entry hits are not marked).

    Default ``view_elev`` / ``view_azim`` match the upright side view used in
    Milestone 5D / source-delta 3D panels (``z`` toward the top of the figure).

    ``max_rays_to_draw`` subsamples successfully refracted rays for readability.
    """
    add_transparent_surface_mesh(
        ax,
        surface_mesh,
        facecolor=mesh_facecolor,
        edgecolor=mesh_edgecolor,
        linewidth=mesh_linewidth,
        alpha=mesh_alpha,
    )

    light_position = _as_xyz(light, ["position"], "light.position")
    plot_point_light(
        ax,
        light_position,
        color=light_color,
        size=light_size,
        label="point light",
    )

    if particle is not None:
        plot_sphere(
            ax,
            particle,
            color=particle_color,
            alpha=particle_alpha,
            filled=True,
            label="particle",
        )

    source_origins = _as_xyz(source_rays, ["origins"], "source_rays.origins")
    parent_indices = np.asarray(refract_result.parent_indices, dtype=int)
    hit_points = np.asarray(refract_result.hit_points, dtype=float)
    valid_mask = np.asarray(refract_result.valid_mask, dtype=bool)

    seg_starts = _as_xyz(segments, ["starts"], "segments.starts")
    seg_ends = _as_xyz(segments, ["ends"], "segments.ends")
    if hasattr(segments, "ray_ids"):
        ray_ids = np.asarray(segments.ray_ids, dtype=int)
    else:
        ray_ids = np.arange(len(seg_starts), dtype=int)

    n_refracted = int(len(parent_indices))
    if n_refracted == 0:
        raise ValueError("refract_result has no successfully refracted rays")

    n_draw = min(int(max_rays_to_draw), n_refracted)
    draw_idx = np.linspace(0, n_refracted - 1, num=n_draw, dtype=int)

    # Map compact refracted-ray index → first matching in-object segment row.
    first_segment_for_ray: dict[int, int] = {}
    for seg_i, rid in enumerate(ray_ids):
        rid_i = int(rid)
        if rid_i not in first_segment_for_ray:
            first_segment_for_ray[rid_i] = int(seg_i)

    collected_points = [light_position.reshape(1, 3)]
    vertices = np.asarray(surface_mesh.vertices, dtype=float)
    collected_points.append(vertices)

    exterior_labeled = False
    interior_labeled = False

    for refracted_i in draw_idx:
        parent = int(parent_indices[refracted_i])
        if not (0 <= parent < len(valid_mask)) or not bool(valid_mask[parent]):
            continue

        origin = source_origins[parent]
        hit = hit_points[parent]
        if not np.all(np.isfinite(hit)):
            continue

        plot_segment(
            ax,
            origin,
            hit,
            color=exterior_color,
            linewidth=exterior_linewidth,
            label="exterior (source→mesh)" if not exterior_labeled else None,
        )
        exterior_labeled = True

        seg_i = first_segment_for_ray.get(int(refracted_i))
        if seg_i is not None:
            plot_segment(
                ax,
                seg_starts[seg_i],
                seg_ends[seg_i],
                color=interior_color,
                linewidth=interior_linewidth,
                label="interior (refracted)" if not interior_labeled else None,
            )
            interior_labeled = True
            collected_points.append(seg_starts[seg_i].reshape(1, 3))
            collected_points.append(seg_ends[seg_i].reshape(1, 3))

        collected_points.append(origin.reshape(1, 3))
        collected_points.append(hit.reshape(1, 3))

    entry_labeled = False
    exit_labeled = False
    if particle_events:
        events = tuple(particle_events)
        if max_particle_events_to_draw is not None:
            events = events[: int(max_particle_events_to_draw)]

        for event in events:
            entry_pt = np.asarray(
                _event_field(event, ["entry_point"], "event.entry_point"),
                dtype=float,
            ).reshape(3)
            exit_pt = np.asarray(
                _event_field(event, ["exit_point"], "event.exit_point"),
                dtype=float,
            ).reshape(3)

            ax.scatter(
                [entry_pt[0]],
                [entry_pt[1]],
                [entry_pt[2]],
                color=particle_entry_color,
                s=particle_point_size,
                depthshade=False,
                label="particle entry" if not entry_labeled else None,
                zorder=7,
            )
            entry_labeled = True

            ax.scatter(
                [exit_pt[0]],
                [exit_pt[1]],
                [exit_pt[2]],
                color=particle_exit_color,
                s=particle_point_size,
                depthshade=False,
                label="particle exit" if not exit_labeled else None,
                zorder=7,
            )
            exit_labeled = True

            collected_points.append(entry_pt.reshape(1, 3))
            collected_points.append(exit_pt.reshape(1, 3))

    if particle is not None:
        center = _as_xyz(particle, ["center"], "particle.center")
        radius = float(
            particle["radius"]
            if isinstance(particle, dict)
            else particle.radius
        )
        collected_points.append((center - radius).reshape(1, 3))
        collected_points.append((center + radius).reshape(1, 3))

    points = np.vstack(collected_points)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    if equal_aspect:
        set_axes_equal(ax)

    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    if title is None:
        title = (
            f"Point-light refraction on phantom "
            f"({n_draw} of {n_refracted} refracted rays)"
        )
    ax.set_title(title)

    if show_legend:
        handles = [
            Line2D([0], [0], color=exterior_color, lw=2, label="exterior (source→mesh)"),
            Line2D([0], [0], color=interior_color, lw=2, label="interior (refracted)"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=light_color,
                markersize=8,
                label="point light",
            ),
        ]
        if particle is not None:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=particle_color,
                    markersize=8,
                    label="particle",
                )
            )
        if entry_labeled:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=particle_entry_color,
                    markersize=6,
                    label="particle entry",
                )
            )
        if exit_labeled:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=particle_exit_color,
                    markersize=6,
                    label="particle exit",
                )
            )
        ax.legend(handles=handles, loc="upper left", fontsize=8)
