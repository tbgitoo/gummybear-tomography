"""3D and 2D matplotlib helpers for particle intersection geometry review."""

from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from mpl_toolkits.mplot3d.axes3d import Axes3D
from trimesh import Trimesh

from ..helpers import event_entry_t, event_exit_t, event_segment_index, get_field, pair_path_id

from gummybear.particles import ParticleSphere

from gummybear.particles import ParticleIntersectionEvent



from typing import Optional, Sequence, Any

import numpy as np




def plot_sphere(
    ax: Axes3D,
    particle: ParticleSphere | dict,
    color: str = "tab:blue",
    alpha: float = 0.6,
    linewidth: float = 0.8,
    label: Optional[str] = None,
) -> None:
    """
    Plot an analytic particle sphere as a 3D wireframe.

    This helper is intended for simple geometry validation purposes.
    The sphere is rendered directly from an analytic sphere representation
    and is not derived from any mesh or tetrahedral discretization.

    The function accepts either:

    - a ParticleSphere-like object with ``particle.center`` and
      ``particle.radius`` attributes,
    - or a dict-like representation with ``particle["center"]`` and
      ``particle["radius"]`` entries.

    This makes the helper robust to ParticleSphere instances as well as
    serialized / dumped / normalized particle records.

    The function is purely a visualization utility and performs no
    optical calculations. It is useful for visually inspecting:

    - particle placement inside the phantom,
    - particle size and relative position,
    - ray-segment / sphere intersection geometry,
    - entry and exit points,
    - midpoint locations used later for particle source deposition.

    Parameters
    ----------
    ax
        Matplotlib 3D axis created with
        ``projection="3d"``.

    particle
        Analytic particle to visualize. May be either a ParticleSphere-like
        object or a dict-like particle record.

    color
        Matplotlib color used for the wireframe.

    alpha
        Wireframe transparency.

    linewidth
        Wireframe line width.

    label
        Optional legend label. If provided, the particle center is
        marked with a scatter point so the object appears in the legend.

    Notes
    -----
    This function is intended for notebook visualization only.
    It is not part of the forward model and does not influence
    ray transport, source deposition, diffusion, or image formation.
    """

    if isinstance(particle, dict):
        try:
            center = np.asarray(particle["center"], dtype=float)
            radius = float(particle["radius"])
        except KeyError as exc:
            raise KeyError(
                "Dict particle must contain keys 'center' and 'radius'. "
                f"Available keys are: {list(particle.keys())}"
            ) from exc
    else:
        try:
            center = np.asarray(particle.center, dtype=float)
            radius = float(particle.radius)
        except AttributeError as exc:
            raise TypeError(
                "particle must be either a ParticleSphere-like object with "
                "attributes '.center' and '.radius', or a dict with keys "
                "'center' and 'radius'. "
                f"Got object of type {type(particle)!r}."
            ) from exc

    if center.shape != (3,):
        raise ValueError(
            f"particle center must be a 3-vector, got shape {center.shape}"
        )

    if radius <= 0:
        raise ValueError(
            f"particle radius must be positive, got {radius}"
        )

    u = np.linspace(0.0, 2.0 * np.pi, 32)
    v = np.linspace(0.0, np.pi, 16)

    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(
        x,
        y,
        z,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
    )

    if label is not None:
        ax.scatter(
            center[0],
            center[1],
            center[2],
            color=color,
            s=50,
            label=label,
        )

def set_axes_equal(ax: Axes3D) -> None:
    """Set equal data limits on x/y/z so spheres appear round in 3D plots.

    Uses the largest axis range among current limits and re-centers all three
    axes on the midpoint. Call after drawing geometry so limits are populated.

    Args:
        ax: Matplotlib 3D axis to adjust.
    """

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])

    radius = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(z_mid - radius, z_mid + radius)

def plot_segment(
    ax,
    start: np.ndarray,
    end: np.ndarray,
    color: str = "black",
    linewidth: float = 2.0,
    label: str | None = None,
) -> None:
    """Draw one straight 3D line segment between two world-space points.

    Args:
        ax: Matplotlib 3D or 2D axis.
        start: Segment start ``(3,)``.
        end: Segment end ``(3,)``.
        color: Matplotlib line color.
        linewidth: Line width in points.
        label: Optional legend label (only the first labeled segment appears).
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        linewidth=linewidth,
        label=label,
    )




def plot_intersection_event(
    ax: Axes3D,
    event: ParticleIntersectionEvent,
    chord_color: str = "tab:red",
    entry_color: str = "tab:green",
    exit_color: str = "tab:blue",
    midpoint_color: str = "tab:orange",
    chord_linewidth: float = 4.0,
    point_size: float = 50.0,
    midpoint_size: float = 80.0,
    label_chord: str | None = None,
) -> None:
    """
    Visualize one particle/segment intersection event in 3D.

    This helper is intended for M5A particle-geometry validation. It draws
    the clipped chord of a ray segment inside one analytic particle sphere,
    together with the entry point, exit point, and chord midpoint.

    The function is purely a visualization helper. It does not modify ray
    intensities, does not perform optical attenuation, does not deposit source
    energy, and does not interact with the diffusion mesh.

    Parameters
    ----------
    ax
        Matplotlib 3D axis created with ``projection="3d"``.

    event
        Particle/segment intersection event containing entry and exit points,
        the path length inside the particle, and the midpoint of the clipped
        chord.

    chord_color
        Color used for the chord inside the particle.

    entry_color
        Color used for the entry point marker.

    exit_color
        Color used for the exit point marker.

    midpoint_color
        Color used for the midpoint marker.

    chord_linewidth
        Line width for the chord inside the particle.

    point_size
        Marker size for entry and exit points.

    midpoint_size
        Marker size for the midpoint marker.

    label_chord
        Optional legend label for the chord. Use this only for the first event
        in a plot to avoid duplicate legend entries.

    Notes
    -----
    The midpoint visualized here is the same geometric midpoint that later M5
    stages may use for local particle delta-source assignment. At the M5A stage,
    it is only shown as a geometric diagnostic.
    """

    plot_segment(
        ax,
        event.entry_point,
        event.exit_point,
        color=chord_color,
        linewidth=chord_linewidth,
        label=label_chord,
    )

    ax.scatter(
        event.entry_point[0],
        event.entry_point[1],
        event.entry_point[2],
        color=entry_color,
        s=point_size,
    )

    ax.scatter(
        event.exit_point[0],
        event.exit_point[1],
        event.exit_point[2],
        color=exit_color,
        s=point_size,
    )

    ax.scatter(
        event.midpoint_inside_particle[0],
        event.midpoint_inside_particle[1],
        event.midpoint_inside_particle[2],
        color=midpoint_color,
        s=midpoint_size,
        marker="x",
    )






def plot_affected_transport_pairs(
    ax: Axes,
    segments: Any,
    ray_ids: np.ndarray,
    pairs: Sequence[Any],
    events: Sequence[Any],
    max_pairs_to_draw: int = 12,
    include_context_paths: bool = True,
    xlabel: str = "distance along ray (mm)",
    title: Optional[str] = None,
    title_fontsize: float = 20.0,
    title_pad: float = 20.0,
    show_legend: bool = True,
) -> None:
    """
    Plot affected transport pairs as a 1D barcode-style ray lineage diagnostic.

    Affected transport paths are drawn as regularly spaced horizontal rows.
    Particle intersection intervals are shown as red sub-intervals with entry,
    midpoint, and exit markers.

    If requested, unaffected context paths are drawn between affected paths and
    interpolated by path ID. This preserves the vertical spacing of affected
    paths while showing nearby unaffected rays.

    Parameters
    ----------
    ax
        Matplotlib 2D axis to draw on.

    segments
        Segment container with ``starts`` and ``ends`` arrays. These may be
        attributes or dict keys.

    ray_ids
        Integer array mapping each segment index to a transport path ID.

    pairs
        Sequence of affected transport pair records. Each pair must expose one
        of ``path_id``, ``ray_id``, or ``transport_path_id`` as an attribute or
        dict key.

    events
        Sequence of particle intersection events. Each event must expose
        ``segment_index``, ``entry_t``, and ``exit_t`` as attributes or dict
        keys.

    max_pairs_to_draw
        Maximum number of affected pairs to draw.

    include_context_paths
        If True, draw unaffected transport paths between affected rows.

    xlabel
        X-axis label.

    title
        Optional plot title. If None, a two-line default title is used.

    title_fontsize
        Font size used for the title.

    title_pad
        Padding between title and axis.

    show_legend
        If True, add a standard legend.

    Notes
    -----
    The y-axis is inverted so path numbering increases from top to bottom.

    Affected paths keep regular spacing:

    ``y = 0, 1, 2, ...``

    Context paths are squeezed between affected rows without changing the
    affected-path row positions.
    """



    starts = np.asarray(
        get_field(segments, ["starts"], "segments.starts"),
        dtype=float,
    )
    ends = np.asarray(
        get_field(segments, ["ends"], "segments.ends"),
        dtype=float,
    )

    ray_ids = np.asarray(ray_ids, dtype=int)
    pairs = tuple(pairs)
    events = tuple(events)

    if len(pairs) == 0:
        raise ValueError("pairs is empty; no affected transport pairs to plot.")

    if len(events) == 0:
        raise ValueError("events is empty; no particle intersections to plot.")

    if len(ray_ids) != len(starts):
        raise ValueError(
            "ray_ids length does not match number of segment starts: "
            f"len(ray_ids)={len(ray_ids)}, len(starts)={len(starts)}"
        )

    if len(starts) != len(ends):
        raise ValueError(
            "segments.starts and segments.ends have different lengths: "
            f"{len(starts)} vs {len(ends)}"
        )

    # ------------------------------------------------------------------
    # Select affected pairs.
    # ------------------------------------------------------------------

    pairs_sorted = sorted(pairs, key=pair_path_id)
    pairs_to_draw = pairs_sorted[:max_pairs_to_draw]

    affected_path_ids_to_draw = np.asarray(
        [pair_path_id(pair) for pair in pairs_to_draw],
        dtype=int,
    )

    if len(affected_path_ids_to_draw) == 0:
        raise ValueError("No affected transport pairs selected for drawing.")

    # ------------------------------------------------------------------
    # Build event lookup by path ID.
    # ------------------------------------------------------------------

    events_by_path_id: dict[int, list[Any]] = {}

    for event in events:
        segment_index = event_segment_index(event)

        if not (0 <= segment_index < len(ray_ids)):
            continue

        path_id = int(ray_ids[segment_index])
        events_by_path_id.setdefault(path_id, []).append(event)

    for path_id in events_by_path_id:
        events_by_path_id[path_id] = sorted(
            events_by_path_id[path_id],
            key=lambda event: (
                event_segment_index(event),
                event_entry_t(event),
            ),
        )

    # ------------------------------------------------------------------
    # Local geometry helpers.
    # ------------------------------------------------------------------

    def segment_length(segment_index: int) -> float:
        start = starts[segment_index]
        end = ends[segment_index]
        return float(np.linalg.norm(end - start))

    def path_segment_indices(path_id: int) -> np.ndarray:
        return np.where(ray_ids == path_id)[0]

    def cumulative_segment_coordinates(segment_indices: np.ndarray):
        """
        Return dictionaries mapping segment index to cumulative start/end
        position along one transport path.
        """

        s0: dict[int, float] = {}
        s1: dict[int, float] = {}

        cursor = 0.0

        for segment_index in segment_indices:
            segment_index = int(segment_index)
            length = segment_length(segment_index)

            s0[segment_index] = cursor
            s1[segment_index] = cursor + length

            cursor += length

        return s0, s1, cursor

    def path_length(path_id: int) -> float:
        segment_indices = path_segment_indices(path_id)
        return sum(segment_length(int(i)) for i in segment_indices)

    # ------------------------------------------------------------------
    # Y-coordinate construction.
    # ------------------------------------------------------------------

    affected_y = np.arange(len(affected_path_ids_to_draw), dtype=float)

    all_path_ids = np.asarray(
        sorted(set(int(path_id) for path_id in ray_ids)),
        dtype=int,
    )

    min_affected_path_id = int(affected_path_ids_to_draw.min())
    max_affected_path_id = int(affected_path_ids_to_draw.max())

    context_path_ids = all_path_ids[
        (all_path_ids >= min_affected_path_id)
        & (all_path_ids <= max_affected_path_id)
    ]

    affected_path_id_set = set(affected_path_ids_to_draw.tolist())

    unaffected_context_path_ids = np.asarray(
        [
            int(path_id)
            for path_id in context_path_ids
            if int(path_id) not in affected_path_id_set
        ],
        dtype=int,
    )

    y_by_path_id: dict[int, float] = {
        int(path_id): float(y)
        for path_id, y in zip(affected_path_ids_to_draw, affected_y)
    }

    if include_context_paths and len(affected_path_ids_to_draw) >= 2:
        unaffected_y = np.interp(
            unaffected_context_path_ids,
            affected_path_ids_to_draw,
            affected_y,
        )

        for path_id, y in zip(unaffected_context_path_ids, unaffected_y):
            y_by_path_id[int(path_id)] = float(y)
    else:
        unaffected_context_path_ids = np.asarray([], dtype=int)

    # ------------------------------------------------------------------
    # X-axis scale and label placement.
    # ------------------------------------------------------------------

    max_path_length = 0.0

    for path_id in context_path_ids:
        max_path_length = max(max_path_length, path_length(int(path_id)))

    x_label_offset = -0.025 * max(max_path_length, 1.0)

    # ------------------------------------------------------------------
    # Draw unaffected context paths first.
    # ------------------------------------------------------------------

    for path_id in unaffected_context_path_ids:
        path_id = int(path_id)
        y = y_by_path_id[path_id]

        segment_indices = path_segment_indices(path_id)

        if len(segment_indices) == 0:
            continue

        _, _, path_len = cumulative_segment_coordinates(segment_indices)

        ax.plot(
            [0.0, path_len],
            [y, y],
            color="0.82",
            linewidth=1.2,
            alpha=0.65,
            solid_capstyle="butt",
            zorder=1,
        )

        ax.text(
            x_label_offset,
            y,
            f"{path_id}",
            ha="right",
            va="center",
            fontsize=7,
            color="0.55",
            zorder=2,
        )

    # ------------------------------------------------------------------
    # Draw affected paths and particle intervals.
    # ------------------------------------------------------------------

    for pair in pairs_to_draw:
        path_id = pair_path_id(pair)
        y = y_by_path_id[path_id]

        segment_indices = path_segment_indices(path_id)

        if len(segment_indices) == 0:
            print(f"WARNING: no segments found for path_id={path_id}")
            continue

        seg_s0, _, path_len = cumulative_segment_coordinates(segment_indices)

        # Clean full affected path baseline.
        ax.plot(
            [0.0, path_len],
            [y, y],
            color="tab:gray",
            linewidth=5.0,
            alpha=0.35,
            solid_capstyle="butt",
            zorder=3,
        )

        # Segment boundaries.
        for segment_index in segment_indices:
            x = seg_s0[int(segment_index)]

            ax.plot(
                [x, x],
                [y - 0.18, y + 0.18],
                color="0.75",
                linewidth=0.6,
                zorder=4,
            )

        ax.plot(
            [path_len, path_len],
            [y - 0.18, y + 0.18],
            color="0.75",
            linewidth=0.6,
            zorder=4,
        )

        # Particle intersections for this path.
        path_events = events_by_path_id.get(path_id, [])

        for event in path_events:
            segment_index = event_segment_index(event)

            if segment_index not in seg_s0:
                continue

            segment_len = segment_length(segment_index)

            x_entry = seg_s0[segment_index] + event_entry_t(event) * segment_len
            x_exit = seg_s0[segment_index] + event_exit_t(event) * segment_len
            x_mid = 0.5 * (x_entry + x_exit)

            ax.plot(
                [x_entry, x_exit],
                [y, y],
                color="tab:red",
                linewidth=5.0,
                solid_capstyle="butt",
                zorder=6,
            )

            ax.scatter(
                x_entry,
                y,
                color="tab:green",
                s=45,
                zorder=7,
            )

            ax.scatter(
                x_mid,
                y,
                color="tab:orange",
                marker="x",
                s=65,
                zorder=8,
            )

            ax.scatter(
                x_exit,
                y,
                color="tab:blue",
                s=45,
                zorder=7,
            )

        # Bold affected path label.
        ax.text(
            x_label_offset,
            y,
            f"path {path_id}",
            ha="right",
            va="center",
            fontsize=9,
            color="black",
            fontweight="bold",
            zorder=9,
        )

    # ------------------------------------------------------------------
    # Formatting.
    # ------------------------------------------------------------------

    ax.set_yticks([])
    ax.set_xlabel(xlabel)

    if title is None:
        title = (
            "Transport Pairs\n"
            f"{len(pairs_to_draw)} affected paths ({len(pairs)} total)"
        )

    ax.set_title(
        title,
        fontsize=title_fontsize,
        pad=title_pad,
    )

    # y=0 at the top, so path numbering increases from top to bottom.
    ax.set_ylim(-0.75, len(pairs_to_draw) - 0.25)
    ax.invert_yaxis()

    ax.set_xlim(x_label_offset * 1.6, max_path_length * 1.02)

    if show_legend:
        legend_handles = [
            Line2D(
                [0], [0],
                color="0.82",
                linewidth=1.5,
                alpha=0.65,
                label="unaffected context path",
            ),
            Line2D(
                [0], [0],
                color="tab:gray",
                linewidth=5.0,
                alpha=0.35,
                label="affected clean transport path",
            ),
            Line2D(
                [0], [0],
                color="tab:red",
                linewidth=5.0,
                label="inside-particle interval",
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor="tab:green",
                markeredgecolor="tab:green",
                markersize=8,
                label="entry point",
            ),
            Line2D(
                [0], [0],
                marker="x",
                color="tab:orange",
                markersize=9,
                linewidth=0,
                label="midpoint",
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor="tab:blue",
                markeredgecolor="tab:blue",
                markersize=8,
                label="exit point",
            ),
        ]

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
        )

    ax.grid(axis="x", alpha=0.25)  





from typing import Any, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_particle_intersections_in_mesh(
    ax: Axes3D,
    surface_mesh: Trimesh,
    particle: Any,
    segments: Any,
    events: Sequence[Any],
    *,
    max_events_to_draw: Optional[int] = 30,
    mesh_facecolor: str = "lightgray",
    mesh_edgecolor: str = "tab:gray",
    mesh_linewidth: float = 0.08,
    mesh_alpha: float = 0.10,
    particle_color: str = "darkgrey",
    particle_alpha: float = 0.55,
    particle_linewidth: float = 1.0,
    full_segment_color: str = "black",
    full_segment_linewidth: float = 1.0,
    chord_color: str = "tab:red",
    entry_color: str = "tab:green",
    exit_color: str = "tab:blue",
    midpoint_color: str = "tab:orange",
    chord_linewidth: float = 1.0,
    point_size: float = 45.0,
    midpoint_size: float = 80.0,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> None:
    """
    Plot particle/segment intersection events inside a transparent surface mesh.

    This helper draws:

    - the transparent phantom surface mesh,
    - the analytic particle sphere,
    - full transport segments that intersect the particle,
    - clipped inside-particle chords,
    - particle entry points,
    - particle exit points,
    - chord midpoints.

    Parameters
    ----------
    ax
        Matplotlib 3D axis created with ``projection="3d"``.

    surface_mesh
        Surface mesh object with ``triangles`` and ``vertices`` attributes.

    particle
        Particle sphere to draw. May be a ParticleSphere-like object or a dict,
        provided ``plot_sphere`` supports both.

    segments
        Segment container with ``starts`` and ``ends`` arrays.

    events
        Sequence of particle intersection events. Each event must expose
        ``segment_index``, ``entry_point``, ``exit_point``, and
        ``midpoint_inside_particle``.

    max_events_to_draw
        Maximum number of events to draw. If None, draw all events.

    mesh_facecolor
        Mesh face color.

    mesh_edgecolor
        Mesh edge color.

    mesh_linewidth
        Mesh edge line width.

    mesh_alpha
        Mesh transparency.

    particle_color
        Particle sphere wireframe color.

    particle_alpha
        Particle sphere transparency.

    particle_linewidth
        Particle sphere wireframe line width.

    full_segment_color
        Color of full intersecting transport segments.

    full_segment_linewidth
        Line width of full intersecting transport segments.

    chord_color
        Color of clipped inside-particle chords.

    entry_color
        Entry marker color.

    exit_color
        Exit marker color.

    midpoint_color
        Midpoint marker color.

    chord_linewidth
        Line width of inside-particle chords.

    point_size
        Marker size for entry and exit points.

    midpoint_size
        Marker size for chord midpoint markers.

    title
        Optional plot title. If None, a default title is generated.

    show_legend
        If True, draw a standard diagnostic legend.

    Notes
    -----
    This function is a visualization helper only. It does not modify transport
    paths, particle properties, attenuation, source deposition, or diffusion
    fields.
    """

    def get_field(obj: Any, names: Sequence[str], label: str) -> Any:
        for name in names:
            if isinstance(obj, dict) and name in obj:
                return obj[name]

            if hasattr(obj, name):
                return getattr(obj, name)

        raise AttributeError(
            f"Could not find {label}. Tried names {list(names)} "
            f"on object of type {type(obj)!r}."
        )

    events = tuple(events)

    if max_events_to_draw is None:
        events_to_draw = events
    else:
        events_to_draw = events[:max_events_to_draw]

    starts = np.asarray(
        get_field(segments, ["starts"], "segments.starts"),
        dtype=float,
    )
    ends = np.asarray(
        get_field(segments, ["ends"], "segments.ends"),
        dtype=float,
    )

    triangles = np.asarray(
        get_field(surface_mesh, ["triangles"], "surface_mesh.triangles"),
        dtype=float,
    )

    vertices = np.asarray(
        get_field(surface_mesh, ["vertices"], "surface_mesh.vertices"),
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Surface mesh
    # ------------------------------------------------------------------

    ax.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolor=mesh_facecolor,
            edgecolor=mesh_edgecolor,
            linewidth=mesh_linewidth,
            alpha=mesh_alpha,
        )
    )

    # ------------------------------------------------------------------
    # Particle sphere
    # ------------------------------------------------------------------

    plot_sphere(
        ax,
        particle,
        color=particle_color,
        alpha=particle_alpha,
        linewidth=particle_linewidth,
        label=None,
    )

    # ------------------------------------------------------------------
    # Full intersecting ray segments
    # ------------------------------------------------------------------

    drawn_segment_indices: set[int] = set()

    for event in events_to_draw:
        segment_index = int(
            get_field(
                event,
                ["segment_index"],
                "event.segment_index",
            )
        )

        if segment_index in drawn_segment_indices:
            continue

        if not (0 <= segment_index < len(starts)):
            continue

        drawn_segment_indices.add(segment_index)

        start = starts[segment_index]
        end = ends[segment_index]

        plot_segment(
            ax,
            start,
            end,
            color=full_segment_color,
            linewidth=full_segment_linewidth,
            label=None,
        )

    # ------------------------------------------------------------------
    # Clipped particle intersection chords
    # ------------------------------------------------------------------

    for event in events_to_draw:
        plot_intersection_event(
            ax,
            event,
            chord_color=chord_color,
            entry_color=entry_color,
            exit_color=exit_color,
            midpoint_color=midpoint_color,
            chord_linewidth=chord_linewidth,
            point_size=point_size,
            midpoint_size=midpoint_size,
            label_chord=None,
        )

    # ------------------------------------------------------------------
    # Axis limits and labels
    # ------------------------------------------------------------------

    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    set_axes_equal(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if title is None:
        title = (
            f"Particle Intersections: "
            f"{len(events_to_draw)} shown / {len(events)} total"
        )

    ax.set_title(title)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    if show_legend:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=full_segment_color,
                linewidth=1.5,
                label="full intersecting ray segment",
            ),
            Line2D(
                [0],
                [0],
                color=chord_color,
                linewidth=chord_linewidth,
                label="inside-particle chord",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=entry_color,
                markeredgecolor=entry_color,
                markersize=8,
                label="particle entry point",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color=midpoint_color,
                markersize=9,
                linewidth=0,
                label="chord midpoint",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=exit_color,
                markeredgecolor=exit_color,
                markersize=8,
                label="particle exit point",
            ),
        ]

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
        )  