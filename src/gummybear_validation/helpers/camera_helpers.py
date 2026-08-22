"""Pinhole camera framing and pose helpers for M2A validation notebooks.

These utilities mirror the basis construction in ``gummybear.rays.camera.make_pinhole_rays``
so validation plots and optional trimesh viewers stay aligned with the raster pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gummybear.rays.camera import PinholeCameraConfig

if TYPE_CHECKING:
    import trimesh


def pinhole_camera_basis(
    cam: PinholeCameraConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return eye position and pinhole ``right`` / ``up`` / ``forward`` unit vectors.

    Uses the same cross-product order as ``make_pinhole_rays`` so projected geometry
    and ray bundles share one camera frame.

    Args:
        cam: Pinhole configuration with ``camera_position``, ``look_at``, and ``up``.

    Returns:
        Tuple ``(eye, forward, right, up)`` as length-3 float arrays.
    """
    eye = np.asarray(cam.camera_position, dtype=float)
    target = np.asarray(cam.look_at, dtype=float)
    up_hint = np.asarray(cam.up, dtype=float)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    return eye, forward, right, up


def pinhole_camera_framing_mesh(
    mesh: trimesh.Trimesh,
    *,
    view_from: tuple[float, float, float] = (0.0, -1.0, 0.0),
    distance_scale: float = 2.35,
    margin: float = 1.16,
    resolution: int = 256,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> PinholeCameraConfig:
    """Build a pinhole config that frames ``mesh`` from outside its volume.

    The camera sits on the ray ``centroid - view_from * distance`` looking at the
    mesh centroid. ``distance`` scales with the mesh bounding-sphere radius so tall
    parts (e.g. ``proto_bear_head.stl``) remain outside the frustum near plane.
    Square FOV is chosen so all front-facing vertices fit on the virtual image plane
    at unit depth, inflated by ``margin``.

    Args:
        mesh: Surface mesh in world/mm coordinates.
        view_from: Direction *from* which the camera looks (normalized internally).
        distance_scale: Multiplier on ``2 * bounding_sphere_radius`` for standoff.
        margin: Extra factor on the projected half-extent when computing FOV.
        resolution: Square raster side length passed through to the config.
        up: World-space up hint for the camera basis (stored on the config).

    Returns:
        ``PinholeCameraConfig`` ready for ``make_camera_rays``.
    """
    center = np.asarray(mesh.centroid, dtype=float)
    view_dir = np.asarray(view_from, dtype=float)
    view_dir /= np.linalg.norm(view_dir)

    span = float(np.max(np.linalg.norm(mesh.vertices - center, axis=1)))
    dist = max(float(distance_scale) * 2.0 * span, 26.0)
    eye = center - view_dir * dist
    target = center

    up_hint = np.asarray(up, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up_v = np.cross(right, forward)
    up_v /= np.linalg.norm(up_v)

    image_dist = 1.0
    rel = np.asarray(mesh.vertices, dtype=float) - eye
    depth = rel @ forward
    front = depth > 1e-3
    rel = rel[front]
    depth = depth[front]
    u = (rel @ right) / depth * image_dist
    v = (rel @ up_v) / depth * image_dist
    half = float(max(u.max(), -u.min(), v.max(), -v.min())) * float(margin)
    fov_deg = float(np.degrees(2.0 * np.arctan(half / image_dist)))

    return PinholeCameraConfig(
        camera_position=tuple(float(x) for x in eye),
        look_at=tuple(float(x) for x in target),
        up=tuple(float(x) for x in up),
        fov_deg=fov_deg,
        resolution=resolution,
    )


def trimesh_camera_transform(cam: PinholeCameraConfig) -> np.ndarray:
    """Camera-to-world pose matrix for trimesh / pyglet viewers.

    Column layout matches trimesh's ``Camera.look_at``: ``X=right``, ``Y=up``,
    ``Z=eye-target`` (away from the look target). OpenGL rendering applies the
    inverse as the view matrix.

    Note:
        trimesh's Jupyter three.js template reads ``gltf.cameras[0]`` and does
        not apply the exported camera-node transform, so ``scene.show()`` in a
        notebook can look rotated relative to the raster. Prefer
        ``scene.show(viewer="gl")`` when pyglet is available.

    Args:
        cam: Pinhole configuration whose basis defines the pose.

    Returns:
        ``(4, 4)`` homogeneous camera-to-world transform.
    """
    eye, _forward, right, up = pinhole_camera_basis(cam)
    away = eye - np.asarray(cam.look_at, dtype=float)
    away /= np.linalg.norm(away)

    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = up
    transform[:3, 2] = away
    transform[:3, 3] = eye
    return transform
