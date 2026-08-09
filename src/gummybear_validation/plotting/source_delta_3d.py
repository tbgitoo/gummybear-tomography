"""3D plotting helpers for particle-induced source-delta fields.

Draws diffusion-element centroids and translucent mesh hulls on an existing 3D
axis. Library functions never create figures or call ``plt.show()``; notebooks
attach colorbars to the returned ``ScalarMappable`` when needed.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def boundary_faces_from_tets(tets):
    """Return exterior triangular faces of a tetrahedral mesh.

    Faces shared by two tets are removed; only boundary faces with count one
    remain.

    Args:
        tets: Integer array of shape ``(T, 4)`` with node indices per tet.

    Returns:
        numpy.ndarray: Boundary faces, shape ``(F, 3)``.
    """
    tets = np.asarray(tets, dtype=int)

    local_faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=int,
    )

    all_faces = tets[:, local_faces].reshape(-1, 3)
    sorted_faces = np.sort(all_faces, axis=1)

    unique_faces, counts = np.unique(
        sorted_faces,
        axis=0,
        return_counts=True,
    )

    return unique_faces[counts == 1]


def add_surface_mesh(
    ax,
    diff_mesh,
    face_alpha=0.13,
    face_color="lightgray",
    edge_color="0.60",
    linewidth=0.20,
):
    """Add a translucent exterior hull of the diffusion mesh to a 3D axis.

    Args:
        ax: Matplotlib 3D axis.
        diff_mesh: Diffusion mesh with ``nodes`` and ``tets``.
        face_alpha: Face transparency.
        face_color: Face fill color.
        edge_color: Edge color for tet boundary lines.
        linewidth: Edge line width.

    Returns:
        mpl_toolkits.mplot3d.art3d.Poly3DCollection: Added surface collection.
    """
    nodes = np.asarray(diff_mesh.nodes, dtype=float)
    boundary_faces = boundary_faces_from_tets(diff_mesh.tets)
    triangles = nodes[boundary_faces]

    surface = Poly3DCollection(
        triangles,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=linewidth,
        alpha=face_alpha,
    )

    ax.add_collection3d(surface)
    return surface


def set_equal_3d_axes(ax, points):
    """Set cubic axis limits centered on ``points`` for isotropic 3D views.

    Uses the maximum span across x/y/z. Calls ``set_box_aspect((1,1,1))`` when
    supported.

    Args:
        ax: Matplotlib 3D axis.
        points: Point cloud used to infer bounds, shape ``(N, 3)``.
    """
    points = np.asarray(points, dtype=float)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def relative_active_mask(values, relative_floor=1e-5, positive_only=False):
    """Mask elements whose magnitude exceeds a scale-relative noise floor.

    The floor is ``relative_floor * max(|values|)`` over candidate nonzero
    entries. When ``positive_only=True``, only positive values participate in
    the scale and threshold.

    Args:
        values: Per-element scalar field.
        relative_floor: Fraction of peak magnitude used as cutoff.
        positive_only: If True, ignore negative values for scale and mask.

    Returns:
        numpy.ndarray: Boolean mask, same shape as ``values``.
    """
    values = np.asarray(values, dtype=float)

    if positive_only:
        candidate = values > 0.0
        scale_values = values[candidate]
    else:
        candidate = np.abs(values) > 0.0
        scale_values = np.abs(values[candidate])

    if scale_values.size == 0:
        return np.zeros(values.shape, dtype=bool)

    floor = float(np.max(scale_values) * relative_floor)

    if positive_only:
        return values > floor

    return np.abs(values) > floor


def projected_far_to_near_order(ax, points):
    """Sort point indices so farther points are drawn before nearer ones.

    Uses Matplotlib's current 3D projection so depth ordering matches the
    fixed camera view.

    Args:
        ax: Matplotlib 3D axis with an initialized view.
        points: World points, shape ``(N, 3)``.

    Returns:
        numpy.ndarray: Integer indices sorting points back-to-front.
    """
    points = np.asarray(points, dtype=float)

    unused_x, unused_y, z_projected = proj3d.proj_transform(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        ax.get_proj(),
    )

    return np.argsort(z_projected)


def scatter_depth_ordered(
    ax,
    points,
    values,
    cmap,
    norm,
    point_size=42,
    alpha=0.95,
):
    """Draw centroid markers from far to near for readable fixed-view 3D plots.

    Each marker is colored individually so depth ordering is stable without
    relying on Matplotlib's default 3D scatter z-sort.

    Args:
        ax: Matplotlib 3D axis.
        points: Centroid positions, shape ``(N, 3)``.
        values: Scalar field used for color mapping, length ``N``.
        cmap: Matplotlib colormap name or object.
        norm: Matplotlib normalizer (for example ``SymLogNorm``).
        point_size: Marker area in points².
        alpha: Marker opacity.
    """
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)

    cmap_obj = plt.get_cmap(cmap)
    order = projected_far_to_near_order(ax, points)

    for i in order:
        rgba = cmap_obj(norm(values[i]))

        ax.scatter(
            points[i, 0],
            points[i, 1],
            points[i, 2],
            color=rgba,
            s=point_size,
            alpha=alpha,
            depthshade=False,
            edgecolors="none",
        )


def source_active_element_mask(
    E_clean_elem,
    E_particle_elem,
    delta_E_background_elem,
    delta_E_particle_scat_elem,
    delta_E_transport_elem,
):
    """Return elements touched by clean, particle, or any source-delta channel.

    An element is active when any input per-element energy array is nonzero
    (exact ``!= 0``; no relative floor).

    Args:
        E_clean_elem: Clean deposited energy per element.
        E_particle_elem: Particle-altered total deposited energy per element.
        delta_E_background_elem: Dirty-minus-clean background delta per element.
        delta_E_particle_scat_elem: Direct particle-scatter delta per element.
        delta_E_transport_elem: Net transport-induced source delta per element.

    Returns:
        numpy.ndarray: Boolean mask, length ``num_elements``.
    """
    E_clean_elem = np.asarray(E_clean_elem, dtype=float)
    E_particle_elem = np.asarray(E_particle_elem, dtype=float)
    delta_E_background_elem = np.asarray(delta_E_background_elem, dtype=float)
    delta_E_particle_scat_elem = np.asarray(delta_E_particle_scat_elem, dtype=float)
    delta_E_transport_elem = np.asarray(delta_E_transport_elem, dtype=float)

    return (
        (np.abs(E_clean_elem) > 0.0)
        | (np.abs(E_particle_elem) > 0.0)
        | (np.abs(delta_E_background_elem) > 0.0)
        | (np.abs(delta_E_particle_scat_elem) > 0.0)
        | (np.abs(delta_E_transport_elem) > 0.0)
    )


def plot_active_element_scalar_3d(
    ax,
    diff_mesh,
    values,
    title,
    active_mask=None,
    signed=True,
    cmap="coolwarm",
    point_size=42,
    relative_floor=1e-5,
    view_elev=24,
    view_azim=-58,
    mesh_face_alpha=0.13,
    mesh_edge_color="0.60",
    mesh_linewidth=0.20,
    mesh_face_color="lightgray",
):
    """Plot active diffusion-element centroids colored by a scalar field.

    Draws a translucent mesh hull, sets an isotropic view, and depth-orders
    centroid markers. Signed fields use ``SymLogNorm``; unsigned positive fields
    use ``LogNorm``. Does not create a figure or colorbar.

    When no elements pass the combined active/value mask, updates the title and
    returns ``None``.

    Args:
        ax: Existing Matplotlib 3D axis.
        diff_mesh: Diffusion mesh with ``nodes``, ``tets``, ``centroids``.
        values: Per-element scalar field.
        title: Base axis title (may gain a suffix when empty).
        active_mask: Optional boolean mask; defaults to all elements.
        signed: If True, allow negative values with symmetric log scaling.
        cmap: Colormap name.
        point_size: Centroid marker size.
        relative_floor: Relative cutoff passed to :func:`relative_active_mask`.
        view_elev: Camera elevation in degrees.
        view_azim: Camera azimuth in degrees.
        mesh_face_alpha: Translucent hull face alpha.
        mesh_edge_color: Hull edge color.
        mesh_linewidth: Hull edge width.
        mesh_face_color: Hull face color.

    Returns:
        matplotlib.cm.ScalarMappable or None: Mappable for ``fig.colorbar``, or
        ``None`` when nothing is plotted.
    """
    values = np.asarray(values, dtype=float)
    centroids = np.asarray(diff_mesh.centroids, dtype=float)

    if active_mask is None:
        active_mask = np.ones(values.shape, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)

    value_mask = relative_active_mask(
        values,
        relative_floor=relative_floor,
        positive_only=not signed,
    )

    mask = active_mask & value_mask

    ax.view_init(elev=view_elev, azim=view_azim)

    add_surface_mesh(
        ax,
        diff_mesh,
        face_alpha=mesh_face_alpha,
        face_color=mesh_face_color,
        edge_color=mesh_edge_color,
        linewidth=mesh_linewidth,
    )

    set_equal_3d_axes(ax, np.asarray(diff_mesh.nodes, dtype=float))

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if not np.any(mask):
        ax.set_title(title + "\n(no active elements)")
        return None

    plotted_values = values[mask]
    plotted_points = centroids[mask]

    if signed:
        abs_values = np.abs(plotted_values)
        nonzero_abs = abs_values[abs_values > 0.0]

        if nonzero_abs.size == 0:
            ax.set_title(title + "\n(no nonzero active elements)")
            return None

        max_abs = float(np.max(nonzero_abs))
        min_abs = float(np.min(nonzero_abs))

        linthresh = max(min_abs, max_abs * relative_floor)

        norm = SymLogNorm(
            linthresh=linthresh,
            vmin=-max_abs,
            vmax=max_abs,
            base=10,
        )
    else:
        positive_values = plotted_values[plotted_values > 0.0]

        if positive_values.size == 0:
            ax.set_title(title + "\n(no positive active elements)")
            return None

        vmin = float(np.min(positive_values))
        vmax = float(np.max(positive_values))

        if vmin == vmax:
            vmin = max(vmax * 0.1, np.finfo(float).tiny)

        norm = LogNorm(vmin=vmin, vmax=vmax)

    scatter_depth_ordered(
        ax,
        plotted_points,
        plotted_values,
        cmap=cmap,
        norm=norm,
        point_size=point_size,
    )

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(plotted_values)

    return mappable


def source_delta_plot_panels(
    delta_E_background_elem,
    delta_E_particle_scat_elem,
    delta_E_transport_elem,
):
    """Return panel specs for the standard three-part source-delta figure.

    Each panel dict contains ``values``, ``title``, ``signed``, ``cmap``, and
    ``colorbar_label`` for shadow (background delta), direct particle scatter,
    and net transport delta respectively.

    Args:
        delta_E_background_elem: Dirty-minus-clean background deposition delta.
        delta_E_particle_scat_elem: Direct particle-scatter source delta.
        delta_E_transport_elem: Net particle-induced source delta.

    Returns:
        list[dict]: Three panel specification dicts ready for plotting loops.
    """
    return [
        {
            "values": np.asarray(delta_E_background_elem, dtype=float),
            "title": "Shadow term\nDirty-minus-clean background deposition",
            "signed": True,
            "cmap": "coolwarm",
            "colorbar_label": "dE background",
        },
        {
            "values": np.asarray(delta_E_particle_scat_elem, dtype=float),
            "title": "Direct particle-scatter source",
            "signed": False,
            "cmap": "viridis",
            "colorbar_label": "dE particle scatter",
        },
        {
            "values": np.asarray(delta_E_transport_elem, dtype=float),
            "title": "Net particle-induced source delta",
            "signed": True,
            "cmap": "coolwarm",
            "colorbar_label": "dE total",
        },
    ]
