"""Light-source geometry: per-face illumination directions and intensity configs."""

from dataclasses import dataclass
from typing import Union

import numpy as np

from ..geometry import normalize_vector, normalize_vectors, face_centroids


@dataclass(frozen=True)
class DirectionalLightConfig:
    """Parallel (collimated) illumination from a distant source.

    ``propagation_direction`` points in the direction light travels, from the
    source side toward the object. All faces receive the same unit direction
    ``b_f = normalize(propagation_direction)``.

    Example: ``propagation_direction=(0, 0, -1)`` means light travels from
    ``+Z`` toward ``-Z``.

    Parameters
    ----------
    propagation_direction:
        3-vector in mesh coordinates (need not be unit length).
    intensity:
        Source strength scale applied uniformly to every face.
    """

    propagation_direction: tuple[float, float, float]
    intensity: float = 1.0


@dataclass(frozen=True)
class PointLightConfig:
    """Isotropic point source at a fixed position in mesh coordinates.

    For each face centroid ``P_f`` the illumination direction is
    ``b_f = normalize(P_f - position)``, pointing from the source toward the
    face. ``intensity`` scales the source; ``falloff`` selects distance
    weighting when converting to per-face or per-ray weights.

    Parameters
    ----------
    position:
        Source location ``(x, y, z)`` in mesh units.
    intensity:
        Source strength at unit distance (falloff-dependent).
    falloff:
        ``"inverse_square"``, ``"linear"``, or ``"none"``.
    r_min:
        Minimum distance clamp for falloff denominators (mesh units).
    """

    position: tuple[float, float, float]
    intensity: float = 1.0
    falloff: str = "inverse_square"
    r_min: float = 1e-6


LightConfig = Union[DirectionalLightConfig, PointLightConfig]


def directional_light_directions_at_faces(mesh, light: DirectionalLightConfig):
    """Return identical per-face illumination directions for a directional light.

    Every row is ``normalize(light.propagation_direction)``.

    Parameters
    ----------
    mesh:
        Triangle mesh (only ``n_faces`` is used).
    light:
        Directional light configuration.

    Returns
    -------
    np.ndarray, shape ``[n_faces, 3]``
        Unit propagation directions ``b_f`` from source side toward object.
    """
    n_faces = len(mesh.faces)

    b = normalize_vector(light.propagation_direction)
    b_f = np.tile(b, (n_faces, 1))

    return b_f


def point_light_directions_at_faces(mesh, light: PointLightConfig):
    """Return per-face directions from a point source toward face centroids.

    ``b_f = normalize(P_f - position)`` for each face centroid ``P_f``.

    Parameters
    ----------
    mesh:
        Triangle mesh providing face centroids.
    light:
        Point light configuration.

    Returns
    -------
    np.ndarray, shape ``[n_faces, 3]``
        Unit propagation directions from source toward each face.
    """
    centroids = face_centroids(mesh)

    source = np.asarray(light.position, dtype=float)
    b_f = normalize_vectors(centroids - source)

    return b_f


def illumination_directions_at_faces(mesh, light: LightConfig):
    """Dispatch per-face illumination propagation directions for any light config.

    Directional lights return the same normalized direction for all faces.
    Point lights return ``normalize(P_f - position)`` per face centroid.

    Parameters
    ----------
    mesh:
        Triangle mesh.
    light:
        :class:`DirectionalLightConfig` or :class:`PointLightConfig`.

    Returns
    -------
    np.ndarray, shape ``[n_faces, 3]``
        Unit illumination directions ``b_f`` pointing from source toward object.

    Raises
    ------
    TypeError
        If ``light`` is neither directional nor point configuration.
    """
    if isinstance(light, DirectionalLightConfig):
        return directional_light_directions_at_faces(mesh, light)

    if isinstance(light, PointLightConfig):
        return point_light_directions_at_faces(mesh, light)

    raise TypeError(f"Unsupported light config type: {type(light)}")
