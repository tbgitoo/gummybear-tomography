"""Matplotlib helpers for Milestone 3 face-transport validation notebooks."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from gummybear.geometry import face_centroids
from gummybear.optics import sample_face_values_to_image
from gummybear.optics.face_transport import FaceOpticalState

from .intersection_geometry import set_axes_equal


def _mesh_bbox_corners(bounds: np.ndarray) -> np.ndarray:
    """Return eight corners of an axis-aligned bounding box ``[min, max]``."""
    bbox_min = np.asarray(bounds[0], dtype=float)
    bbox_max = np.asarray(bounds[1], dtype=float)
    return np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
        ],
        dtype=float,
    )


def plot_source_rays_with_bbox(
    source_rays,
    mesh_bounds: np.ndarray,
    *,
    quiver_length: float = 10.0,
    figsize: tuple[float, float] = (8.0, 8.0),
) -> None:
    """Plot parallel source rays and the mesh bounding box in 3D."""
    origins = np.asarray(source_rays.origins, dtype=float)
    directions = np.asarray(source_rays.directions, dtype=float)
    corners = _mesh_bbox_corners(mesh_bounds)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    ax.quiver(
        origins[:, 0],
        origins[:, 1],
        origins[:, 2],
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        length=quiver_length,
    )
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color="black",
            linewidth=2,
        )

    all_points = np.vstack([origins, corners])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    plt.show()


def plot_face_state_on_mesh_3d(
    mesh,
    state: FaceOpticalState,
    *,
    title: str = "face energy on mesh",
    figsize: tuple[float, float] = (8.0, 8.0),
    point_size: float = 6.0,
    show_mesh_hull: bool = False,
    mesh_alpha: float = 0.03,
) -> None:
    """Scatter face centroids colored by ``state.face_energy`` where valid."""
    covered = np.asarray(state.valid, dtype=bool)
    if not np.any(covered):
        raise ValueError("state has no valid faces to plot")

    centroids = face_centroids(mesh)[covered]
    values = state.face_energy[covered]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if show_mesh_hull:
        vertices = np.asarray(mesh.vertices, dtype=float)
        ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            s=0.2,
            alpha=mesh_alpha,
        )
    sc = ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c=values,
        cmap="viridis",
        s=point_size,
    )
    plt.colorbar(sc, ax=ax, label="face energy")
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def sample_face_state_camera_images(
    state: FaceOpticalState,
    hit_faces: np.ndarray,
    sample_shape: tuple[int, int],
    *,
    face_areas: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sample energy, coverage, b_out components, and optional density."""
    hit_faces = np.asarray(hit_faces)
    images: dict[str, np.ndarray] = {}

    def _sample(values: np.ndarray) -> np.ndarray:
        img = sample_face_values_to_image(hit_faces, values, background_value=0.0)
        return img.reshape(sample_shape)

    images["energy"] = _sample(state.face_energy)
    images["coverage"] = _sample(state.hit_count.astype(float))
    images["b_out_x"] = _sample(state.b_out[:, 0])
    images["b_out_y"] = _sample(state.b_out[:, 1])
    images["b_out_z"] = _sample(state.b_out[:, 2])

    if face_areas is not None:
        density = np.divide(
            state.face_energy,
            np.asarray(face_areas, dtype=float),
            out=np.zeros_like(state.face_energy),
            where=np.asarray(face_areas) > 0,
        )
        images["energy_density"] = _sample(density)

    return images


def plot_face_state_camera_panel(
    images: dict[str, np.ndarray],
    *,
    title_prefix: str = "",
    show_raw_energy: bool = True,
    show_energy_density: bool = False,
    figsize: tuple[float, float] = (14.0, 8.0),
) -> None:
    """Plot the standard 2×3 M3 camera diagnostic panel from sampled images.

    Stage 1 layout (default): energy, log energy, coverage | b_out x/y/z.
    Stage 3: set ``show_raw_energy=False``, ``show_energy_density=True``.
    """
    top: list[tuple[str, np.ndarray, dict]] = []
    if show_raw_energy:
        top.append(("energy", images["energy"], {}))
    top.append(("log energy", np.log1p(images["energy"]), {}))
    if show_energy_density:
        if "energy_density" not in images:
            raise KeyError("energy_density required when show_energy_density=True")
        top.append(("log energy density", np.log1p(images["energy_density"]), {}))
    top.append(("coverage", images["coverage"], {}))
    top = top[:3]

    bottom = [
        ("b_out x", images["b_out_x"], {"vmin": -1, "vmax": 1}),
        ("b_out y", images["b_out_y"], {"vmin": -1, "vmax": 1}),
        ("b_out z", images["b_out_z"], {"vmin": -1, "vmax": 1}),
    ]
    panels = top + bottom

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    prefix = f"{title_prefix} " if title_prefix else ""
    for ax, (name, img, kwargs) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(img, origin="lower", **kwargs)
        ax.set_title(prefix + name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()


def plot_entry_internal_directions_3d(
    mesh,
    entry_points: np.ndarray,
    internal_dirs: np.ndarray,
    internal_valid: np.ndarray,
    *,
    max_arrows: int = 300,
    figsize: tuple[float, float] = (8.0, 8.0),
) -> None:
    """Plot a subsample of refracted internal directions at entry points."""
    valid_ids = np.flatnonzero(internal_valid)
    if valid_ids.size == 0:
        raise ValueError("no valid internal directions to plot")
    step = max(1, len(valid_ids) // max_arrows)
    sample_ids = valid_ids[::step]

    pts = np.asarray(entry_points[sample_ids], dtype=float)
    dirs = np.asarray(internal_dirs[sample_ids], dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.2, alpha=0.03)
    ax.quiver(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        dirs[:, 0],
        dirs[:, 1],
        dirs[:, 2],
        length=1.5,
        normalize=True,
        color="red",
    )
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    ax.set_title("Refracted internal directions at entry points")
    fig.tight_layout()
    plt.show()
