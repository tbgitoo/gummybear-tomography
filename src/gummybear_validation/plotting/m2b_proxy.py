"""Diagnostic plots for the M2B transient translucent-camera proxy."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from gummybear.optics import (
    CameraSampledBeamVectorField,
    beam_view_coupling,
    sample_face_values_to_image,
)


def collect_m2b_proxy_diagnostics(
    field: CameraSampledBeamVectorField,
    L_proxy: np.ndarray,
    T_face: np.ndarray,
    face_source_intensity: np.ndarray,
    I_proxy: np.ndarray,
) -> dict[str, np.ndarray]:
    """Gather camera-sampled factors for M2B factor-decomposition plots.

    Uses ``beam_view_coupling`` from ``gummybear.optics`` so notebook panels
    match ``beam_vector_field_to_camera_image(..., mode="bdotv")``.

    Args:
        field: Durable camera-sampled beam vector field.
        L_proxy: Per-face upstream thickness proxy.
        T_face: Per-face Beer–Lambert transmittance.
        face_source_intensity: Per-face source strength.
        I_proxy: Composed provisional intensity image.

    Returns:
        Dict with keys ``valid_mask``, ``L_sampled``, ``T_sampled``,
        ``source_sampled``, ``g_bv``, and ``I_plot`` (invalid pixels NaN).
    """
    L_sampled = sample_face_values_to_image(
        field.hit_faces,
        L_proxy,
        background_value=np.nan,
    )
    T_sampled = sample_face_values_to_image(
        field.hit_faces,
        T_face,
        background_value=np.nan,
    )
    source_sampled = sample_face_values_to_image(
        field.hit_faces,
        face_source_intensity,
        background_value=np.nan,
    )

    g_bv = beam_view_coupling(field).astype(float)
    g_bv[~field.valid_mask] = np.nan

    I_plot = I_proxy.astype(float).copy()
    I_plot[~field.valid_mask] = np.nan

    return {
        "valid_mask": field.valid_mask.astype(float),
        "L_sampled": L_sampled,
        "T_sampled": T_sampled,
        "source_sampled": source_sampled,
        "g_bv": g_bv,
        "I_plot": I_plot,
    }


def plot_m2b_proxy_diagnostics(
    field: CameraSampledBeamVectorField,
    L_proxy: np.ndarray,
    T_face: np.ndarray,
    face_source_intensity: np.ndarray,
    I_proxy: np.ndarray,
    *,
    show_beam_components: Literal["partial", "full", "none"] = "partial",
) -> None:
    """Plot the standard 2×4 M2B factor panel (+ optional beam component row).

    Args:
        field: Camera-sampled beam vector field from ``camera_sampled_beam_vector_field``.
        L_proxy: Per-face upstream thickness proxy.
        T_face: Per-face transmittance.
        face_source_intensity: Per-face source intensity.
        I_proxy: Composed ``I_proxy`` image.
        show_beam_components: ``"partial"`` fills the 2×4 grid with beam x/y;
            ``"full"`` adds a separate 1×3 beam x/y/z row; ``"none"`` omits beam panels.
    """
    diag = collect_m2b_proxy_diagnostics(
        field,
        L_proxy,
        T_face,
        face_source_intensity,
        I_proxy,
    )

    scalar_items = [
        (diag["valid_mask"], "valid mask"),
        (diag["L_sampled"], "L_proxy sampled"),
        (diag["T_sampled"], "T_face sampled"),
        (diag["source_sampled"], "source intensity sampled"),
        (diag["g_bv"], "g(beam · observation)"),
        (diag["I_plot"], "I_proxy = source · T · g"),
    ]

    if show_beam_components == "none":
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, (img, title) in zip(axes.ravel(), scalar_items, strict=True):
            im = ax.imshow(img, origin="lower")
            ax.set_title(title)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for ax, (img, title) in zip(axes.ravel()[:6], scalar_items, strict=True):
        im = ax.imshow(img, origin="lower")
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_beam_components == "partial":
        beam = field.beam_directions
        beam_items = [
            (beam[..., 0], "beam x"),
            (beam[..., 1], "beam y"),
        ]
        for ax, (img, title) in zip(axes.ravel()[6:], beam_items, strict=True):
            img_plot = img.astype(float).copy()
            img_plot[~field.valid_mask] = np.nan
            im = ax.imshow(img_plot, origin="lower")
            ax.set_title(title)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()

    if show_beam_components == "full":
        plot_beam_direction_components(field)


def plot_beam_direction_components(field: CameraSampledBeamVectorField) -> None:
    """Plot beam direction x/y/z components masked to valid camera pixels."""
    beam = field.beam_directions
    plot_face_sampled_row_from_pixels(
        [beam[..., 0], beam[..., 1], beam[..., 2]],
        ["beam x", "beam y", "beam z"],
        valid_mask=field.valid_mask,
    )


def plot_face_sampled_row_from_pixels(
    images: list[np.ndarray],
    titles: list[str],
    *,
    valid_mask: np.ndarray,
    figsize: tuple[float, float] = (15.0, 4.0),
) -> None:
    """Like ``plot_face_sampled_row`` but masks invalid pixels to NaN."""
    masked = []
    for img in images:
        img_plot = img.astype(float).copy()
        img_plot[~valid_mask] = np.nan
        masked.append(img_plot)

    from gummybear_validation.plotting.face_sampling import plot_face_sampled_row

    plot_face_sampled_row(masked, titles, figsize=figsize)
