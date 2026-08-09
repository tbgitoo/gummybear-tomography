"""Sample face-level beam directions onto camera rays (geometry only, no intensity)."""

from typing import Union

import numpy as np
import trimesh
from dataclasses import dataclass

from ..rays import CameraRayBundle, first_visible_hits
from .light_source import (
    DirectionalLightConfig,
    LightConfig,
    PointLightConfig,
    illumination_directions_at_faces,
)
from ..geometry import face_centroids


@dataclass(frozen=True)
class CameraSampledBeamVectorField:
    """Beam and observation direction fields indexed by camera hit faces.

    Holds per-pixel beam directions (illumination or refracted outgoing ``b_out``)
    and observation directions (mesh toward camera), together with first-surface
    ``hit_faces`` from a camera ray bundle. No intensity, transmittance, or
    thickness is computed here.

    Attributes
    ----------
    ray_bundle:
        Source camera rays defining ``sample_shape`` and pixel geometry.
    beam_directions:
        Per-pixel beam vectors, shape ``[H, W, 3]``; zero outside ``valid_mask``.
    observation_directions:
        Per-pixel view vectors (typically ``-ray_bundle.directions_field``),
        shape ``[H, W, 3]``.
    hit_faces:
        First visible face index per pixel, shape ``[H, W]``; ``-1`` on miss.
    valid_mask:
        Boolean mask of rays that hit the mesh, shape ``[H, W]``.
    """

    ray_bundle: CameraRayBundle

    beam_directions: np.ndarray          # [H, W, 3]
    observation_directions: np.ndarray   # [H, W, 3], usually -ray_bundle.directions_field
    hit_faces: np.ndarray                # [H, W], -1 for miss
    valid_mask: np.ndarray               # [H, W]

    def __post_init__(self):
        H, W = self.ray_bundle.sample_shape

        if self.beam_directions.shape != (H, W, 3):
            raise ValueError(
                f"Expected beam_directions shape {(H, W, 3)}, "
                f"got {self.beam_directions.shape}"
            )

        if self.observation_directions.shape != (H, W, 3):
            raise ValueError(
                f"Expected observation_directions shape {(H, W, 3)}, "
                f"got {self.observation_directions.shape}"
            )

        if self.hit_faces.shape != (H, W):
            raise ValueError(
                f"Expected hit_faces shape {(H, W)}, got {self.hit_faces.shape}"
            )

        if self.valid_mask.shape != (H, W):
            raise ValueError(
                f"Expected valid_mask shape {(H, W)}, got {self.valid_mask.shape}"
            )

    @property
    def sample_shape(self) -> tuple[int, int]:
        """Camera image height and width ``(H, W)``."""
        return self.ray_bundle.sample_shape


def source_intensity_at_faces(
    mesh: trimesh.Trimesh,
    light: Union[DirectionalLightConfig, PointLightConfig],
) -> np.ndarray:
    """Return per-face source intensity factor before transmittance or coupling.

    Directional lights assign constant ``light.intensity`` to every face.
    Point lights apply distance falloff from ``light.position`` to each face
    centroid ``P_f`` with distance ``r_f = ||P_f - position||`` clamped by
    ``light.r_min``:

    - ``falloff="inverse_square"``: ``I_f = intensity / max(r_f^2, r_min^2)``
    - ``falloff="linear"``: ``I_f = intensity / max(r_f, r_min)``
    - ``falloff="none"``: ``I_f = intensity``

    Parameters
    ----------
    mesh:
        Triangle mesh providing face centroids.
    light:
        Directional or point light configuration.

    Returns
    -------
    np.ndarray, shape ``[n_faces]``
        Source intensity factor per face (linear intensity units, not normalized).

    Raises
    ------
    ValueError
        Unknown point-light ``falloff`` mode.
    TypeError
        Unsupported ``light`` type.
    """
    n_faces = len(mesh.faces)

    if isinstance(light, DirectionalLightConfig):
        return np.full(n_faces, light.intensity, dtype=float)

    if isinstance(light, PointLightConfig):
        centroids = face_centroids(mesh)
        source = np.asarray(light.position, dtype=float)

        r = np.linalg.norm(centroids - source, axis=1)
        r_safe = np.maximum(r, light.r_min)

        if light.falloff == "none":
            return np.full(n_faces, light.intensity, dtype=float)

        if light.falloff == "linear":
            return light.intensity / r_safe

        if light.falloff == "inverse_square":
            return light.intensity / (r_safe**2)

        raise ValueError(
            f"Unknown point-light falloff mode: {light.falloff!r}. "
            "Expected 'inverse_square', 'linear', or 'none'."
        )

    raise TypeError(f"Unsupported light config type: {type(light)}")


def sample_face_vectors_to_camera(
    hit_faces: np.ndarray,
    face_vectors: np.ndarray,
    ray_bundle: CameraRayBundle,
    background_vector: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Sample per-face vectors onto the camera sampling grid."""
    H, W = ray_bundle.sample_shape

    hit_faces = np.asarray(hit_faces, dtype=int).reshape(H, W)
    face_vectors = np.asarray(face_vectors, dtype=float)

    if face_vectors.ndim != 2 or face_vectors.shape[1] != 3:
        raise ValueError(
            f"Expected face_vectors shape [n_faces, 3], got {face_vectors.shape}"
        )

    sampled = np.empty((H, W, 3), dtype=float)
    sampled[...] = np.asarray(background_vector, dtype=float)

    valid = (
        (hit_faces >= 0)
        & (hit_faces < len(face_vectors))
    )

    sampled[valid] = face_vectors[hit_faces[valid]]

    return sampled


def camera_sampled_beam_vector_field(
    ray_bundle: CameraRayBundle,
    mesh: trimesh.Trimesh,
    light: LightConfig,
) -> CameraSampledBeamVectorField:
    """Sample face-level beam directions onto camera first-surface hits.

    Casts ``ray_bundle`` through ``first_visible_hits``, gathers per-face beam
    directions from :func:`~gummybear.optics.light_source.illumination_directions_at_faces`,
    and forms observation directions as ``-ray_bundle.directions_field``.
    Missed rays zero out beam and observation vectors.

    No intensity, upstream thickness, or transmittance is computed. Refractive
    pipelines should replace face beam directions with refracted outgoing
    ``b_out[f]`` before camera sampling or coupling.

    Parameters
    ----------
    ray_bundle:
        Camera rays with ``sample_shape`` ``(H, W)``.
    mesh:
        Surface mesh for visibility and face directions.
    light:
        Directional or point light supplying ``b_f``.

    Returns
    -------
    CameraSampledBeamVectorField
        Beam and observation vector fields plus ``hit_faces`` / ``valid_mask``.

    Notebook / protocol:
        Beam sampling for translucent camera proxy (M2B); M3 may substitute refracted ``b_out``.
    """
    H, W = ray_bundle.sample_shape

    hits = first_visible_hits(
        mesh,
        ray_bundle
    )

    hit_faces = np.asarray(hits[2], dtype=int).reshape(H, W)
    valid_mask = np.asarray(hits[0], dtype=bool).reshape(H, W)

    face_beam_directions = illumination_directions_at_faces(mesh, light)

    beam_directions = sample_face_vectors_to_camera(
        hit_faces=hit_faces,
        face_vectors=face_beam_directions,
        ray_bundle=ray_bundle,
        background_vector=(0.0, 0.0, 0.0),
    )

    # Camera ray direction is camera -> mesh.
    # Observation direction is mesh -> camera.
    observation_directions = -ray_bundle.directions_field.copy()

    beam_directions[~valid_mask] = 0.0
    observation_directions[~valid_mask] = 0.0

    return CameraSampledBeamVectorField(
        ray_bundle=ray_bundle,
        beam_directions=beam_directions,
        observation_directions=observation_directions,
        hit_faces=hit_faces,
        valid_mask=valid_mask,
    )
