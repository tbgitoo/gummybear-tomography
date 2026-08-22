"""Face→pixel sampling plots for M2B validation notebooks."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from gummybear.geometry import face_centroids
from gummybear.optics import sample_face_values_to_image


def face_centroid_channels(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-face X/Y/Z centroid scalars and face indices.

    Args:
        mesh: ``trimesh.Trimesh`` with triangular faces.

    Returns:
        Tuple ``(x_face, y_face, z_face, i_face)`` each shape ``[n_faces]``.
    """
    centroids = face_centroids(mesh)
    i_face = np.arange(len(mesh.faces), dtype=float)
    return centroids[:, 0], centroids[:, 1], centroids[:, 2], i_face


def sample_face_scalars_to_images(
    hit_faces: np.ndarray,
    face_scalars: Sequence[np.ndarray],
    sample_shape: tuple[int, int],
    *,
    background_value: float = np.nan,
) -> list[np.ndarray]:
    """Sample multiple face-indexed scalars onto the camera raster.

    Args:
        hit_faces: First-surface face index per pixel, shape ``[H, W]``.
        face_scalars: One length-``n_faces`` array per channel.
        sample_shape: ``(H, W)`` raster shape from the ray bundle.
        background_value: Fill for missed rays in ``sample_face_values_to_image``.

    Returns:
        List of 2-D images reshaped to ``sample_shape``.
    """
    images: list[np.ndarray] = []
    for values in face_scalars:
        img = sample_face_values_to_image(
            hit_faces,
            values,
            background_value=background_value,
        )
        images.append(img.reshape(sample_shape))
    return images


def plot_face_sampled_row(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    figsize: tuple[float, float] = (15.0, 4.0),
    origin: str = "lower",
) -> None:
    """Plot a horizontal row of face-sampled scalar images with color bars.

    Args:
        images: 2-D arrays to display (one panel each).
        titles: Panel titles, same length as ``images``.
        figsize: Matplotlib figure size in inches.
        origin: Passed to ``imshow`` (``"lower"`` matches camera row-major).
    """
    if len(images) != len(titles):
        raise ValueError("images and titles must have the same length")

    ncols = len(images)
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles, strict=True):
        im = ax.imshow(img, origin=origin)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()
