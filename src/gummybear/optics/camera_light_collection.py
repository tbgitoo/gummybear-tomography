"""Camera-side beam–view coupling and provisional translucent intensity composition."""

from ..optics.face_illumination import sample_face_values_to_image

from .illumination_pass import CameraSampledBeamVectorField

import numpy as np


def beam_view_coupling(
    field: CameraSampledBeamVectorField,
) -> np.ndarray:
    """Compute hemispherical cosine coupling between beam and observation directions.

    For each pixel, unit-normalizes ``field.beam_directions`` and
    ``field.observation_directions``, then returns
    ``g = 0.5 * (cos(b · v) + 1)`` clipped to ``[-1, 1]``.

    This is a provisional detector coupling, not a full BRDF or sensor model.

    Parameters
    ----------
    field:
        Camera-sampled beam and observation vector fields on shape ``[H, W, 3]``.

    Returns
    -------
    np.ndarray, shape ``[H, W]``
        Coupling factor in ``[0, 1]`` per pixel.

    Notebook / protocol:
        M2B ``bdotv`` mode in :func:`beam_vector_field_to_camera_image`.
    """
    b = field.beam_directions
    v = field.observation_directions

    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)

    b = b / np.maximum(b_norm, 1e-12)
    v = v / np.maximum(v_norm, 1e-12)

    cos_theta = np.sum(b * v, axis=-1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return 0.5 * (cos_theta + 1.0)


def beam_vector_field_to_camera_image(
    field: CameraSampledBeamVectorField,
    T_face: np.ndarray,
    face_source_intensity: np.ndarray,
    I_bg: float = 1.0,
    mode: str = "bdotv",
) -> np.ndarray:
    """Form a provisional translucent-camera intensity from face-level factors.

    Samples per-face transmittance ``T_face`` and source intensity onto camera
    ``hit_faces``, multiplies by optional beam–view coupling ``g``, and scales
    by ``I_bg``:

        I = I_bg * source_img * T_img * g

    Invalid camera rays (``~field.valid_mask``) are forced to zero. This is an
    approximate translucent-camera proxy, not a validated detector or transport model.

    Parameters
    ----------
    field:
        Camera-sampled beam vectors and hit-face indices.
    T_face:
        Per-face transmittance, shape ``[n_faces]`` (typically from Beer–Lambert).
    face_source_intensity:
        Per-face source strength, shape ``[n_faces]``.
    I_bg:
        Global background / exposure scale.
    mode:
        ``"bdotv"`` applies :func:`beam_view_coupling`; ``"none"`` uses ``g=1``.

    Returns
    -------
    np.ndarray, shape ``[H, W]``
        Provisional camera intensity image.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"bdotv"`` or ``"none"``.

    Notebook / protocol:
        Translucent camera proxy (M2B) before refractive transport (M3).
    """
    T_img = sample_face_values_to_image(
        field.hit_faces,
        T_face,
        background_value=0.0,
    )

    source_img = sample_face_values_to_image(
        field.hit_faces,
        face_source_intensity,
        background_value=0.0,
    )

    if mode == "none":
        g = np.ones_like(T_img)

    elif mode == "bdotv":
        g = beam_view_coupling(field)

    else:
        raise ValueError(
            f"Unknown mode {mode!r}; expected 'none' or 'bdotv'"
        )

    I = I_bg * source_img * T_img * g

    I[~field.valid_mask] = 0.0

    return I
