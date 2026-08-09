# optics/face_illumination.py

import numpy as np


def sample_face_values_to_image(
    hit_faces,
    face_values,
    background_value=-1.0,
):
    """Scatter per-face scalars onto a camera hit-face grid.

    Each pixel takes ``face_values[hit_faces]`` when ``hit_faces >= 0``;
    misses and invalid indices receive ``background_value``.

    Parameters
    ----------
    hit_faces:
        Integer face index per pixel, same shape as the camera image
        (typically ``[H, W]``). Use ``-1`` for rays that miss the mesh.
    face_values:
        Per-face scalar field, shape ``[n_faces]``.
    background_value:
        Fill value for misses and invalid face indices.

    Returns
    -------
    np.ndarray
        Sampled image with the same shape as ``hit_faces``, dtype ``float``.
    """
    image = np.full(
        hit_faces.shape,
        background_value,
        dtype=float,
    )

    valid = hit_faces >= 0

    image[valid] = face_values[
        hit_faces[valid]
    ]

    return image
