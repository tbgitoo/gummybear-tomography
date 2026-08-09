"""Camera models and ray-bundle construction for image sampling."""

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass(frozen=True)
class OrthographicCameraConfig:
    """Parallel (orthographic) camera placed outside the mesh looking inward.

    Rays share a common direction from the camera position toward ``look_at``.
    The image plane spans a square of side ``size`` in world units, centered on
    the look target in the camera's right/up frame.

    Attributes
    ----------
    camera_position : tuple[float, float, float]
        World-space camera origin in mesh coordinates (mm by convention).
    look_at : tuple[float, float, float]
        Point the camera forward axis passes through (often the mesh centroid).
    size : float
        Physical width and height of the orthographic view frustum in mm.
    resolution : int
        Square image side length ``H == W``.
    up : tuple[float, float, float]
        Hint vector used to build the camera's right/up basis; need not be
        orthogonal to the view direction.
    margin : float
        Fraction of bounding-box extent added beyond half-size along the view
        axis when auto-framing (used by higher-level setup helpers).
    """

    camera_position: tuple[float, float, float] = (0.0, -20.0, 0.0)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: float = 20
    resolution: int = 256
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    margin: float = 0.15


@dataclass(frozen=True)
class PinholeCameraConfig:
    """Perspective camera with a square field of view.

    Rays originate at ``camera_position`` and fan out through a virtual image
    plane one unit in front of the camera. The camera must sit outside the
    mesh volume so primary rays can reach the surface.

    Attributes
    ----------
    camera_position : tuple[float, float, float]
        World-space pinhole location in mesh coordinates (mm by convention).
    look_at : tuple[float, float, float]
        Point the central ray passes through (often the mesh centroid).
    up : tuple[float, float, float]
        Hint vector used to build the camera's right/up basis.
    fov_deg : float
        Full horizontal/vertical field of view in degrees (square sensor).
    resolution : int
        Square image side length ``H == W``.
    """

    camera_position: tuple[float, float, float] = (0.0, -30.0, 0.0)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_deg: float = 60.0
    resolution: int = 256


@dataclass(frozen=True)
class CameraRayBundle:
    """Flat camera rays plus the image grid shape used to reshape them.

    Stores parallel ``origins`` and ``directions`` arrays with one row per
    pixel. Directions follow the camera-to-scene convention (from the camera
    toward the mesh).

    Attributes
    ----------
    origins : np.ndarray, shape (N, 3)
        Ray start points in world/mesh coordinates.
    directions : np.ndarray, shape (N, 3)
        Unit or unnormalized view directions; downstream intersection code
        re-normalizes as needed.
    sample_shape : tuple[int, int]
        Image grid shape ``(H, W)`` with ``N == H * W``.
    """

    origins: np.ndarray
    directions: np.ndarray
    sample_shape: tuple[int, int]

    def __post_init__(self):
        origins = np.asarray(self.origins)
        directions = np.asarray(self.directions)

        if origins.shape != directions.shape:
            raise ValueError(
                f"origins shape {origins.shape} does not match "
                f"directions shape {directions.shape}"
            )

        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError(
                f"Expected origins/directions shape [N, 3], got {origins.shape}"
            )

        H, W = self.sample_shape
        if origins.shape[0] != H * W:
            raise ValueError(
                f"Ray count {origins.shape[0]} does not match "
                f"sample_shape {self.sample_shape} = {H * W}"
            )

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return ``sample_shape`` as ``(H, W)``."""
        return self.sample_shape

    @property
    def vector_field_shape(self) -> tuple[int, int, int]:
        """Return the per-pixel vector field shape ``(H, W, 3)``."""
        H, W = self.sample_shape
        return (H, W, 3)

    @property
    def directions_field(self) -> np.ndarray:
        """Return directions reshaped to ``(H, W, 3)``."""
        H, W = self.sample_shape
        return self.directions.reshape(H, W, 3)

    @property
    def origins_field(self) -> np.ndarray:
        """Return origins reshaped to ``(H, W, 3)``."""
        H, W = self.sample_shape
        return self.origins.reshape(H, W, 3)


CameraConfig = Union[OrthographicCameraConfig, PinholeCameraConfig]


def make_orthographic_rays(
    cam: OrthographicCameraConfig,
) -> CameraRayBundle:
    """Build parallel camera rays on a square orthographic grid.

    Parameters
    ----------
    cam : OrthographicCameraConfig
        Camera placement, view size, and resolution.

    Returns
    -------
    CameraRayBundle
        Flat bundle with ``sample_shape=(resolution, resolution)``.
    """
    camera_pos = np.asarray(cam.camera_position, dtype=float)
    look_at = np.asarray(cam.look_at, dtype=float)
    up_hint = np.asarray(cam.up, dtype=float)

    # camera forward direction
    forward = look_at - camera_pos
    forward /= np.linalg.norm(forward)

    # camera right vector
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)

    # corrected orthogonal up vector
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # physical image size in mm
    extent = cam.size
    res = cam.resolution

    xs = np.linspace(-extent / 2, extent / 2, res)
    ys = np.linspace(-extent / 2, extent / 2, res)

    X, Y = np.meshgrid(xs, ys)

    origins = (
        camera_pos
        + X.ravel()[:, None] * right
        + Y.ravel()[:, None] * up
    )

    directions = np.tile(forward, (origins.shape[0], 1))

    return CameraRayBundle(
        origins=origins,
        directions=directions,
        sample_shape=(res, res),
    )


def make_pinhole_rays(
    cam: PinholeCameraConfig,
) -> CameraRayBundle:
    """Build perspective camera rays through a square pinhole frustum.

    Parameters
    ----------
    cam : PinholeCameraConfig
        Camera placement, field of view, and resolution.

    Returns
    -------
    CameraRayBundle
        Flat bundle with ``sample_shape=(resolution, resolution)``.
    """
    camera_pos = np.asarray(cam.camera_position, dtype=float)
    look_at = np.asarray(cam.look_at, dtype=float)
    up_hint = np.asarray(cam.up, dtype=float)

    # camera basis
    forward = look_at - camera_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # image plane
    res = cam.resolution
    fov_rad = np.deg2rad(cam.fov_deg)

    # image plane located 1 unit in front of camera
    image_dist = 1.0
    half_size = np.tan(fov_rad / 2.0) * image_dist

    xs = np.linspace(-half_size, half_size, res)
    ys = np.linspace(-half_size, half_size, res)

    X, Y = np.meshgrid(xs, ys)

    image_points = (
        camera_pos
        + image_dist * forward
        + X.ravel()[:, None] * right
        + Y.ravel()[:, None] * up
    )

    origins = np.tile(camera_pos, (image_points.shape[0], 1))

    directions = image_points - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    return CameraRayBundle(
        origins=origins,
        directions=directions,
        sample_shape=(res, res),
    )


def make_camera_rays(camera: CameraConfig) -> CameraRayBundle:
    """Dispatch to the orthographic or pinhole ray builder.

    Parameters
    ----------
    camera : OrthographicCameraConfig or PinholeCameraConfig
        Camera configuration instance.

    Returns
    -------
    CameraRayBundle
        Flat ray bundle for the configured model.

    Raises
    ------
    TypeError
        When ``camera`` is neither orthographic nor pinhole config.
    """
    if isinstance(camera, OrthographicCameraConfig):
        return make_orthographic_rays(camera)

    if isinstance(camera, PinholeCameraConfig):
        return make_pinhole_rays(camera)

    raise TypeError(f"Unsupported camera config type: {type(camera)}")
