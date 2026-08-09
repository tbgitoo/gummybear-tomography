"""Compose refractive direct and volumetric diffuse camera channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


FORWARD_MODEL_TIER = "m4_refractive_diffusion"
"""Forward-model tier tag for M4 hybrid direct-plus-diffuse composition."""


@dataclass(frozen=True)
class HybridImageResult:
    """Hybrid camera image with separate direct, diffuse, and total channels.

    ``I_total = alpha * I_direct + I_diffuse`` with optional masking and
    negative clipping applied to ``I_total``. ``alpha`` is an explicit relative
    scale on the direct channel and must be recorded in experiment metadata.

    Attributes
    ----------
    I_direct, I_diffuse, I_total:
        Composited images, same shape (typically ``[H, W]``).
    alpha:
        Direct-channel scale factor (dimensionless relative weight).
    forward_model:
        Tier identifier (default :data:`FORWARD_MODEL_TIER`).
    metadata:
        Composition formula and component flags.
    """

    I_direct: np.ndarray
    I_diffuse: np.ndarray
    I_total: np.ndarray
    alpha: float
    forward_model: str = FORWARD_MODEL_TIER
    metadata: dict[str, Any] = field(default_factory=dict)


def compose_hybrid_image(
    I_direct: np.ndarray,
    I_diffuse: np.ndarray,
    *,
    alpha: float = 1.0,
    camera_mask: np.ndarray | None = None,
    clip_negative: bool = True,
) -> HybridImageResult:
    """Combine direct and diffuse camera channels into a total intensity image.

    Forms ``I_total = alpha * I_direct + I_diffuse``. Volumetric diffusion adds to the
    refractive direct channel; it does not replace it. ``alpha`` scales the refractive
    direct contribution only — record it explicitly when exporting manifests or
    model artifacts.

    When ``camera_mask`` is provided, all three outputs are zero outside the
    mask. Optionally clips ``I_total`` to be non-negative.

    Parameters
    ----------
    I_direct:
        Refractive direct channel from :func:`~gummybear.optics.face_transport.compute_refractive_direct_image`.
    I_diffuse:
        Diffuse channel from :func:`~gummybear.optics.diffuse_sampling.sample_diffuse_image`
        (fluence field ``Phi`` sampled at camera hits).
    alpha:
        Relative direct-channel scale (default ``1.0``).
    camera_mask:
        Optional boolean mask matching image shape.
    clip_negative:
        If ``True``, clamp ``I_total`` to ``>= 0``.

    Returns
    -------
    HybridImageResult

    Raises
    ------
    ValueError
        Shape mismatch between inputs or mask.

    Notebook / protocol:
        M4E ``I_total = alpha * I_direct + I_diffuse`` hybrid forward model.
    """
    I_direct = np.asarray(I_direct, dtype=float)
    I_diffuse = np.asarray(I_diffuse, dtype=float)
    if I_direct.shape != I_diffuse.shape:
        raise ValueError(
            f"I_direct shape {I_direct.shape} != I_diffuse shape {I_diffuse.shape}"
        )
    alpha_v = float(alpha)
    I_total = alpha_v * I_direct + I_diffuse
    if camera_mask is not None:
        mask = np.asarray(camera_mask, dtype=bool)
        if mask.shape != I_total.shape:
            raise ValueError("camera_mask shape must match images")
        I_direct = np.where(mask, I_direct, 0.0)
        I_diffuse = np.where(mask, I_diffuse, 0.0)
        I_total = np.where(mask, I_total, 0.0)
    if clip_negative:
        I_total = np.maximum(I_total, 0.0)

    return HybridImageResult(
        I_direct=I_direct,
        I_diffuse=I_diffuse,
        I_total=I_total,
        alpha=alpha_v,
        forward_model=FORWARD_MODEL_TIER,
        metadata={
            "alpha": alpha_v,
            "composition": "I_total = alpha * I_direct + I_diffuse",
            "forward_model": FORWARD_MODEL_TIER,
            "components": {
                "direct_refractive": True,
                "volumetric_diffusion": True,
                "particle_perturbation": False,
            },
        },
    )
