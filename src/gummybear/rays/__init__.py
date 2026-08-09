"""Ray-tracing utilities for camera and source bundles.

The public ray API is re-exported here so callers can write::

    from gummybear.rays import CameraRayBundle, make_camera_rays, first_visible_hits
"""

from .camera import (
    CameraConfig,
    CameraRayBundle,
    OrthographicCameraConfig,
    PinholeCameraConfig,
    make_camera_rays,
    make_orthographic_rays,
    make_pinhole_rays,
)
from .source import RayBundleProtocol, SourceRayBundle
from .visibility import (
    first_visible_hits,
    first_visible_hits_with_points,
    hits_to_image,
    simple_camera_intensity,
    normalize_direction_sums,
)


__all__ = [
    "CameraConfig",
    "CameraRayBundle",
    "OrthographicCameraConfig",
    "PinholeCameraConfig",
    "make_camera_rays",
    "make_orthographic_rays",
    "make_pinhole_rays",
    "RayBundleProtocol",
    "SourceRayBundle",
    "first_visible_hits",
    "first_visible_hits_with_points",
    "hits_to_image",
    "simple_camera_intensity",
    "normalize_direction_sums",
]
