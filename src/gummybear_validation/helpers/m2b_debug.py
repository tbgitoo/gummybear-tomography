"""M2B transient proxy pipeline helpers for validation notebooks.

These wrap ``gummybear.optics`` face-level Beer–Lambert scaffolding used only
for Milestone 2B debugging — not sequence generation or ML physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gummybear.optics import (
    CameraSampledBeamVectorField,
    beam_vector_field_to_camera_image,
    camera_sampled_beam_vector_field,
    compute_face_upstream_thickness,
    source_intensity_at_faces,
    thickness_to_transmittance,
)
from gummybear.optics.light_source import LightConfig

if TYPE_CHECKING:
    import trimesh

    from gummybear.rays import CameraRayBundle


@dataclass(frozen=True)
class M2BDebugProxyResult:
    """Face-level and camera-sampled outputs from the M2B debug compose stack."""

    L_proxy: np.ndarray
    T_face: np.ndarray
    face_source_intensity: np.ndarray
    field: CameraSampledBeamVectorField
    I_proxy: np.ndarray


def compute_m2b_debug_proxy(
    mesh: trimesh.Trimesh,
    ray_bundle: CameraRayBundle,
    light: LightConfig,
    *,
    mu: float = 0.15,
    eps: float = 1e-4,
    I_bg: float = 1.0,
    mode: str = "bdotv",
) -> M2BDebugProxyResult:
    """Run the full M2B transient proxy pipeline (steps 4–6 in plan §Evidence).

    Computes upstream half-ray thickness ``L_proxy``, Beer–Lambert ``T_face``,
    per-face source intensity, the durable ``camera_sampled_beam_vector_field``,
    and provisional ``I_proxy`` via ``beam_vector_field_to_camera_image``.

    Args:
        mesh: Surface mesh in world/mm coordinates.
        ray_bundle: Pinhole camera rays with ``sample_shape`` ``(H, W)``.
        light: Directional or point light configuration.
        mu: Attenuation coefficient for ``thickness_to_transmittance``.
        eps: Probe offset for upstream thickness rays.
        I_bg: Background scale passed to image composition.
        mode: ``"bdotv"`` or ``"none"`` for beam–view coupling.

    Returns:
        ``M2BDebugProxyResult`` with all intermediate face and image fields.
    """
    L_proxy = compute_face_upstream_thickness(mesh, light, eps=eps)
    T_face = thickness_to_transmittance(L_proxy, mu=mu)
    face_source_intensity = source_intensity_at_faces(mesh, light)
    field = camera_sampled_beam_vector_field(ray_bundle, mesh, light)
    I_proxy = beam_vector_field_to_camera_image(
        field=field,
        T_face=T_face,
        face_source_intensity=face_source_intensity,
        I_bg=I_bg,
        mode=mode,
    )
    return M2BDebugProxyResult(
        L_proxy=L_proxy,
        T_face=T_face,
        face_source_intensity=face_source_intensity,
        field=field,
        I_proxy=I_proxy,
    )


def print_m2b_face_fields(
    L_proxy: np.ndarray,
    T_face: np.ndarray,
    face_source_intensity: np.ndarray,
) -> None:
    """Print min/max/mean summaries for transient M2B face-level arrays."""
    from gummybear_validation.helpers.access_helpers import summarize_array

    summarize_array("L_proxy", L_proxy)
    summarize_array("T_face", T_face)
    summarize_array("face_source_intensity", face_source_intensity)
