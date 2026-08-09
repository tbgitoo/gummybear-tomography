"""Validation plotting helpers for element / transport-pair energy deposition."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gummybear.particles import deposit_transport_pair_sources
from gummybear.particles.access_helpers import get_any, pair_path_id


def plot_element_scalar(
    ax,
    diff_mesh,
    values,
    *,
    title,
    cmap="coolwarm",
    symmetric=False,
    point_size=18,
):
    """Scatter diffusion-element centroids colored by a per-element scalar.

    Values are sampled at tetrahedral centroids on a 3D Matplotlib axis.
    When ``symmetric=True``, the color scale is centered at zero with equal
    positive/negative limits; otherwise limits span ``[min, max]``.

    Args:
        ax: Matplotlib 3D axis (``projection="3d"``).
        diff_mesh: Diffusion mesh exposing ``centroids`` with shape ``(E, 3)``.
        values: Per-element scalar field, length ``E``.
        title: Axis title string.
        cmap: Matplotlib colormap name.
        symmetric: If True, use a diverging scale around zero.
        point_size: Marker size for centroid scatter points.

    Returns:
        matplotlib.collections.PathCollection: Scatter artist for colorbars.
    """
    values = np.asarray(values, dtype=float)
    centroids = np.asarray(diff_mesh.centroids, dtype=float)

    if symmetric:
        vmax = float(np.max(np.abs(values))) if values.size else 1.0
        vmin = -vmax
    else:
        vmin = float(np.min(values)) if values.size else 0.0
        vmax = float(np.max(values)) if values.size else 1.0

    scatter = ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c=values,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return scatter


def _as_sequence(x):
    if x is None:
        return ()
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return tuple(x)


def _events_by_path_id(events, ray_ids):
    result = {}
    for event in events:
        segment_index = int(event.segment_index)
        path_id = int(ray_ids[segment_index])
        result.setdefault(path_id, []).append(event)
    return result


def _interval_point(interval, which):
    """Read start/end from a TransportInterval or an asdict()-style mapping."""
    names = (
        ("start", "point0", "start_point", "p0")
        if which == "start"
        else ("end", "point1", "end_point", "p1")
    )
    if isinstance(interval, dict):
        for name in names:
            if name in interval and interval[name] is not None:
                return np.asarray(interval[name], dtype=float)
    else:
        for name in names:
            if hasattr(interval, name):
                value = getattr(interval, name)
                if value is not None:
                    return np.asarray(value, dtype=float)
    raise AttributeError(
        f"Could not read interval {which} point from {type(interval)!r}"
    )


def _path_direction_from_intervals(intervals):
    if not intervals:
        return np.zeros(3), np.array([1.0, 0.0, 0.0])
    p_start = _interval_point(intervals[0], "start")
    p_end = _interval_point(intervals[-1], "end")
    direction = p_end - p_start
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        direction = direction / norm
    return p_start, direction


def _element_order_along_path(diff_mesh, elem_indices, origin, direction):
    centroids = np.asarray(diff_mesh.centroids, dtype=float)
    s = centroids[elem_indices] @ direction - origin @ direction
    order = np.argsort(s)
    return elem_indices[order], s[order]


def _event_s_coordinate(event, origin, direction, which):
    if which == "entry":
        point = np.asarray(event.entry_point, dtype=float)
    elif which == "exit":
        point = np.asarray(event.exit_point, dtype=float)
    else:
        raise ValueError(which)
    return float(point @ direction - origin @ direction)


def _nearest_bar_x_for_s(s_value, elem_s_values):
    if len(elem_s_values) == 0:
        return None
    return int(np.argmin(np.abs(elem_s_values - s_value)))


def plot_transport_pair_deposition_barcodes(
    pairs,
    segments,
    events,
    ray_ids,
    diff_mesh,
    *,
    mu_s,
    mu_a,
    max_rays=10,
    seed=0,
    random_select=True,
    density=True,
    figsize_per_ray=1.3,
    log_scale=False,
    log_floor_fraction=1e-6,
):
    """Plot clean versus particle-altered source deposition along affected rays.

    For each selected transport pair, deposits source using
    ``deposit_transport_pair_sources`` (dirty background plus attenuated-chord
    particle scatter—the validated M5C assignment). Elements are ordered along
    the clean path direction; particle entry/exit markers align to the nearest
    deposited element.

    Requires at least one affected pair. Raises when deposition arrays are
    missing or a pair lacks ``clean_intervals``.

    Args:
        pairs: Affected transport pair records.
        segments: Segment bundle (unused; geometry comes from pair intervals).
        events: Particle intersection events indexed by segment.
        ray_ids: Segment → path id mapping.
        diff_mesh: Diffusion mesh with ``centroids``, ``volumes``.
        mu_s: Scattering coefficient for deposition.
        mu_a: Absorption coefficient for deposition.
        max_rays: Cap on number of pair subplots.
        seed: RNG seed when ``random_select=True``.
        random_select: If True, sample pairs randomly; else take the first N.
        density: If True, divide deposited energy by element volume.
        figsize_per_ray: Vertical inches per pair subplot.
        log_scale: If True, plot totals on a log axis with a small floor.
        log_floor_fraction: Multiplier for positive max when building log floor.

    Returns:
        tuple[matplotlib.figure.Figure, list]: Figure and per-pair axes.
    """
    del segments  # pair intervals already carry geometry / intensity
    ray_ids = np.asarray(ray_ids, dtype=int)
    pairs = tuple(pairs)

    if not pairs:
        raise ValueError("No affected transport pairs to plot.")

    rng = np.random.default_rng(seed)

    if len(pairs) > max_rays:
        if random_select:
            selected_indices = np.sort(
                rng.choice(len(pairs), size=max_rays, replace=False)
            )
        else:
            selected_indices = np.arange(max_rays)
    else:
        selected_indices = np.arange(len(pairs))

    selected_pairs = [pairs[i] for i in selected_indices]
    events_by_path = _events_by_path_id(events, ray_ids)

    fig, axes = plt.subplots(
        nrows=len(selected_pairs),
        ncols=1,
        figsize=(14, figsize_per_ray * len(selected_pairs) + 2.0),
        sharex=False,
    )

    if len(selected_pairs) == 1:
        axes = [axes]

    volumes = np.asarray(diff_mesh.volumes, dtype=float)

    for ax, pair in zip(axes, selected_pairs):
        path_id = pair_path_id(pair)
        # Prefer live dataclass fields. require_any/asdict would turn nested
        # TransportInterval objects into plain dicts.
        clean_intervals = _as_sequence(
            getattr(pair, "clean_intervals", None)
            or get_any(pair, ["clean_intervals", "clean_transport", "clean"])
        )
        if not clean_intervals:
            raise AssertionError(f"path {path_id}: missing clean_intervals")

        deposition = deposit_transport_pair_sources(
            diff_mesh,
            pair,
            mu_s=mu_s,
            mu_a=mu_a,
            assignment="attenuated_chord",
        )

        clean_E = deposition.E_clean_elem
        dirty_background_E = deposition.E_dirty_background_elem
        dirty_particle_E = deposition.E_particle_scat_elem
        dirty_total_E = deposition.E_dirty_total_elem

        if density:
            clean_values = clean_E / volumes
            dirty_background_values = dirty_background_E / volumes
            dirty_particle_values = dirty_particle_E / volumes
            dirty_total_values = dirty_total_E / volumes
            ylabel = "source density"
        else:
            clean_values = clean_E
            dirty_background_values = dirty_background_E
            dirty_particle_values = dirty_particle_E
            dirty_total_values = dirty_total_E
            ylabel = "scattered energy"

        delta_values = dirty_total_values - clean_values

        elem_indices = np.asarray(
            sorted(
                set(np.nonzero(clean_values)[0].tolist())
                | set(np.nonzero(dirty_total_values)[0].tolist())
            ),
            dtype=int,
        )

        if len(elem_indices) == 0:
            ax.set_title(f"path {path_id}: no deposited elements")
            continue

        origin, direction = _path_direction_from_intervals(clean_intervals)
        elem_indices, elem_s_values = _element_order_along_path(
            diff_mesh,
            elem_indices,
            origin,
            direction,
        )

        x = np.arange(len(elem_indices))
        width = 0.36

        clean_ordered = clean_values[elem_indices]
        dirty_background_ordered = dirty_background_values[elem_indices]
        dirty_particle_ordered = dirty_particle_values[elem_indices]
        dirty_total_ordered = dirty_total_values[elem_indices]
        delta_ordered = delta_values[elem_indices]

        if not log_scale:
            ax.bar(
                x - width / 2,
                clean_ordered,
                width=width,
                color="tab:gray",
                alpha=0.75,
                label="clean background" if ax is axes[0] else None,
            )
            ax.bar(
                x + width / 2,
                dirty_background_ordered,
                width=width,
                color="tab:blue",
                alpha=0.75,
                label="dirty background" if ax is axes[0] else None,
            )
            ax.bar(
                x + width / 2,
                dirty_particle_ordered,
                bottom=dirty_background_ordered,
                width=width,
                color="tab:orange",
                alpha=0.85,
                label="dirty particle scatter" if ax is axes[0] else None,
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.plot(
                x,
                delta_ordered,
                color="black",
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                label="dirty-clean delta" if ax is axes[0] else None,
            )
        else:
            positive_values = np.concatenate(
                [
                    clean_ordered[clean_ordered > 0],
                    dirty_total_ordered[dirty_total_ordered > 0],
                ]
            )
            if len(positive_values):
                log_floor = float(np.max(positive_values) * log_floor_fraction)
                log_floor = max(log_floor, np.finfo(float).tiny)
            else:
                log_floor = np.finfo(float).tiny

            clean_plot = np.where(clean_ordered > 0, clean_ordered, log_floor)
            dirty_plot = np.where(
                dirty_total_ordered > 0, dirty_total_ordered, log_floor
            )

            ax.bar(
                x - width / 2,
                clean_plot,
                width=width,
                color="tab:gray",
                alpha=0.75,
                label="clean total" if ax is axes[0] else None,
            )
            ax.bar(
                x + width / 2,
                dirty_plot,
                width=width,
                color="tab:purple",
                alpha=0.75,
                label="dirty total" if ax is axes[0] else None,
            )
            ax.set_yscale("log")

        path_events = events_by_path.get(int(path_id), [])
        for event in path_events:
            entry_s = _event_s_coordinate(event, origin, direction, "entry")
            exit_s = _event_s_coordinate(event, origin, direction, "exit")
            entry_x = _nearest_bar_x_for_s(entry_s, elem_s_values)
            exit_x = _nearest_bar_x_for_s(exit_s, elem_s_values)
            if entry_x is not None:
                ax.axvline(
                    entry_x,
                    color="tab:green",
                    linestyle=":",
                    linewidth=1.4,
                    label="particle entry" if ax is axes[0] else None,
                )
            if exit_x is not None:
                ax.axvline(
                    exit_x,
                    color="tab:red",
                    linestyle=":",
                    linewidth=1.4,
                    label="particle exit" if ax is axes[0] else None,
                )

        scale_label = "log scale" if log_scale else "linear scale"
        ax.text(
            0.01,
            0.92,
            f"path {path_id}  |  start → end  |  {scale_label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in elem_indices], rotation=90, fontsize=7)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("diffusion element index, ordered along transport path")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=4 if log_scale else 5,
        frameon=True,
    )
    scale_title = "Log Scale" if log_scale else "Linear Scale"
    fig.suptitle(
        "Per-Ray Clean/Particle-Altered Source Deposition\n"
        f"{scale_title} | {len(selected_pairs)} affected transport paths shown",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, axes
