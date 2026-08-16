"""Assemble a POV-Ray 3.7 scene for an M8 physical setup."""

from dataclasses import dataclass

import numpy as np
import trimesh

from gummybear.paths import display_path

from .load_sample import PhysicalSetup
from .pov_primitives import box, comment_block, cone, cylinder, plane_z, pov_vec, sphere


@dataclass(frozen=True)
class IllustrationCameraParams:
    """POV-Ray *illustration* viewpoint (not the M8 acquisition pose).

    ``yaw_deg`` is measured from behind the acquisition pinhole toward the
    side overview (0° looks along the optical axis from behind the camera).
    ``fov_deg`` is the POV-Ray camera ``angle``.
    ``distance_scale`` multiplies the pinhole–bear distance to set how far
    behind the pinhole the viewpoint sits (larger = further back).
    """

    yaw_deg: float = 60.0
    fov_deg: float = 46.0
    distance_scale: float = 1.58


def _basis(forward: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = forward / np.linalg.norm(forward)
    right = np.cross(f, up_hint)
    n = float(np.linalg.norm(right))
    if n < 1e-9:
        right = np.cross(f, np.array([1.0, 0.0, 0.0]))
        n = float(np.linalg.norm(right))
    right = right / n
    up = np.cross(right, f)
    up = up / np.linalg.norm(up)
    return f, right, up


def _scale_along_ray(
    origin: np.ndarray,
    point: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Keep direction ``origin → point``, multiply distance by ``scale``."""
    origin = np.asarray(origin, dtype=float).reshape(3)
    point = np.asarray(point, dtype=float).reshape(3)
    vec = point - origin
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return point.copy()
    s = float(scale)
    if s <= 0.0:
        raise ValueError(f"illustration distance scale must be > 0, got {s}")
    return origin + (vec / dist) * (dist * s)


def illustrated_acquisition_position(
    setup: PhysicalSetup, *, distance_scale: float = 1.0
) -> np.ndarray:
    """Catalog pinhole moved along the look-at ray (illustration only)."""
    return _scale_along_ray(
        setup.camera_look_at, setup.camera_position, distance_scale
    )


def illustrated_catalog_light_position(
    setup: PhysicalSetup,
    mesh_bounds: np.ndarray,
    *,
    distance_scale: float = 1.0,
) -> np.ndarray:
    """Catalog light moved along the AABB-centre ray (illustration only)."""
    center = np.asarray(mesh_bounds, dtype=float).mean(axis=0)
    return _scale_along_ray(center, setup.light_position, distance_scale)


def _rotate_xy_about_z(point: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate ``(x, y)`` about the world z-axis; keep ``z``."""
    p = np.asarray(point, dtype=float).reshape(3).copy()
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    x, y = float(p[0]), float(p[1])
    p[0] = c * x - s * y
    p[1] = s * x + c * y
    return p


def camera_frustum_segments(
    setup: PhysicalSetup,
    *,
    depth_frac: float = 0.35,
    origin: np.ndarray | None = None,
    look_at: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Four frustum edges from pinhole through image-plane corners.

    Uses real ``camera_position``, ``look_at``, ``up``, and ``fov_deg``.
    These are **not** stored pixel rays; they visualize the pinhole FOV.
    """
    origin = (
        np.asarray(setup.camera_position, dtype=float)
        if origin is None
        else np.asarray(origin, dtype=float)
    )
    look = (
        np.asarray(setup.camera_look_at, dtype=float)
        if look_at is None
        else np.asarray(look_at, dtype=float)
    )
    up_vec = (
        np.asarray(setup.camera_up, dtype=float)
        if up is None
        else np.asarray(up, dtype=float)
    )
    forward, right, up_b = _basis(look - origin, up_vec)
    dist = float(np.linalg.norm(look - origin))
    depth = max(dist * float(depth_frac), 1.0)
    half = np.tan(np.deg2rad(setup.camera_fov_deg) / 2.0) * depth
    corners = [
        origin + depth * forward + sx * half * right + sy * half * up_b
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return tuple((origin.copy(), c) for c in corners)


def _angle_delta_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def nearest_anomaly_png(
    plates: dict[float, str],
    angle_deg: float,
    *,
    atol_deg: float = 0.51,
) -> str | None:
    """Return the POV basename whose frame angle is closest, or None."""
    if not plates:
        return None
    angle = float(angle_deg)
    best_ang = None
    best_d = None
    for cand in plates:
        d = _angle_delta_deg(cand, angle)
        if best_d is None or d < best_d:
            best_d = d
            best_ang = cand
    if best_d is None or best_d > float(atol_deg):
        return None
    return plates[float(best_ang)]


def _back_anomaly_plate(png_name: str, half: float, body: float) -> str:
    """Single UV-mapped quad, same size as the camera-body back face.

    Stored images: columns increase with camera-right, row 0 is camera-up
    (after ``orient_camera_image_for_storage``). POV ``uv_mapping`` puts
    ``(0,0)`` at the image lower-left. The plate is viewed from behind,
    looking along +forward, so u increases toward local −Y (−right) would
    mirror the scene; u=0 is therefore at local −right.
    """
    back = -(half + 0.015 * body)
    sy = half * 0.7
    sz = half * 0.7
    return (
        "  mesh2 {\n"
        "    vertex_vectors { 4,\n"
        f"      <{back:.6g}, {sy:.6g}, {-sz:.6g}>,\n"
        f"      <{back:.6g}, {-sy:.6g}, {-sz:.6g}>,\n"
        f"      <{back:.6g}, {-sy:.6g}, {sz:.6g}>,\n"
        f"      <{back:.6g}, {sy:.6g}, {sz:.6g}>\n"
        "    }\n"
        "    uv_vectors { 4, <1,0>, <0,0>, <0,1>, <1,1> }\n"
        "    face_indices { 2, <0,1,2>, <0,2,3> }\n"
        "    uv_indices { 2, <0,1,2>, <0,2,3> }\n"
        "    double_illuminate\n"
        "    texture {\n"
        "      uv_mapping\n"
        f'      pigment {{ image_map {{ png "{png_name}" once interpolate 2 }} }}\n'
        "      finish { ambient 1.0 diffuse 0.0 }\n"
        "    }\n"
        "  }\n"
    )


def _camera_pigment(rgb: tuple[float, float, float], transmit: float) -> str:
    r, g, b = rgb
    t = float(transmit)
    if t <= 1e-6:
        return f"rgb <{r:g}, {g:g}, {b:g}>"
    t = min(max(t, 0.0), 0.94)
    return f"rgbt <{r:g}, {g:g}, {b:g}, {t:.4g}>"


def acquisition_camera_solids(
    setup: PhysicalSetup,
    *,
    origin: np.ndarray | None = None,
    look_at: np.ndarray | None = None,
    up: np.ndarray | None = None,
    transmit: float = 0.0,
    header_comment: str | None = None,
    anomaly_png: str | None = None,
) -> str:
    """Box body + cylinder lens at an optical pinhole (scene object)."""
    origin = (
        np.asarray(setup.camera_position, dtype=float)
        if origin is None
        else np.asarray(origin, dtype=float)
    )
    look = (
        np.asarray(setup.camera_look_at, dtype=float)
        if look_at is None
        else np.asarray(look_at, dtype=float)
    )
    up_vec = (
        np.asarray(setup.camera_up, dtype=float)
        if up is None
        else np.asarray(up, dtype=float)
    )
    forward, right, up_b = _basis(look - origin, up_vec)
    dist = float(np.linalg.norm(look - origin))
    body = max(dist * 0.055, 3.2)
    half = body * 0.45
    body_p = _camera_pigment((0.18, 0.18, 0.2), transmit)
    lens_p = _camera_pigment((0.12, 0.12, 0.14), transmit)
    if header_comment is None:
        header_comment = (
            "Acquisition camera (M8 look-at and angle from the manifest). "
            "Distance may be illustration-scaled. Neutral gray. "
            "This is a scene marker, not the POV-Ray render camera."
        )
    parts = [
        comment_block(header_comment) if header_comment else "",
        "union {\n",
        "  box { "
        f"<-{half:.4g}, -{half * 0.7:.4g}, -{half * 0.7:.4g}>, "
        f"<{half:.4g}, {half * 0.7:.4g}, {half * 0.7:.4g}>\n"
        f"    pigment {{ {body_p} }}\n"
        "    finish { phong 0.15 ambient 0.06 }\n"
        "  }\n",
        "  cylinder { "
        f"<0, 0, 0>, <{body * 0.55:.4g}, 0, 0>, {half * 0.28:.4g}\n"
        f"    pigment {{ {lens_p} }}\n"
        "    finish { phong 0.2 }\n"
        "  }\n",
    ]
    if anomaly_png:
        parts.append(_back_anomaly_plate(anomaly_png, half, body))
    parts.extend(
        [
        f"  matrix < {forward[0]:.6g}, {right[0]:.6g}, {up_b[0]:.6g},\n"
        f"           {forward[1]:.6g}, {right[1]:.6g}, {up_b[1]:.6g},\n"
        f"           {forward[2]:.6g}, {right[2]:.6g}, {up_b[2]:.6g},\n"
        f"           {origin[0]:.6g}, {origin[1]:.6g}, {origin[2]:.6g} >\n"
        "  no_shadow\n"
        "}\n",
        ]
    )
    return "".join(parts)


def tomography_orbit_cameras(
    setup: PhysicalSetup,
    *,
    origin: np.ndarray,
    n_each_side: int,
    step_deg: float,
    frustum_radius: float,
    draw_frustum: bool = True,
    anomaly_plates: dict[float, str] | None = None,
    fade: bool = True,
) -> str:
    """Ghost optical cameras on a z-orbit, fading with angular step.

    Pinholes share the illustrated camera's distance from the world z-axis
    and its z. Look-at stays the catalog look-at. Illustration only.
    """
    n = int(n_each_side)
    if n < 0:
        raise ValueError(f"orbit camera count must be >= 0, got {n}")
    if n == 0:
        return comment_block("Tomography orbit cameras omitted (count 0).")
    step = float(step_deg)
    if step <= 0.0:
        raise ValueError(f"orbit step_deg must be > 0, got {step}")
    origin = np.asarray(origin, dtype=float).reshape(3)
    look = np.asarray(setup.camera_look_at, dtype=float)
    up0 = np.asarray(setup.camera_up, dtype=float)
    fade_note = (
        "Transmit increases with |k|."
        if fade
        else "Orbit fade off: cameras are opaque like the main optical camera."
    )
    bits = [
        comment_block(
            "Tomography orbit cameras: same |xy| from world z as the illustrated "
            f"optical pinhole, rotated by k×{step:g}°. {fade_note} "
            "Look-at is the catalog look-at. Not extra acquisition frames."
        )
    ]
    if draw_frustum:
        bits.append(comment_block("Orbit camera frustum edges (camera_rays=all)."))
    for k in (*range(-n, 0), *range(1, n + 1)):
        angle = k * step
        abs_angle = float(setup.camera_angle_deg) + angle
        transmit = (0.32 + 0.58 * (abs(k) / n)) if fade else 0.0
        pos = _rotate_xy_about_z(origin, angle)
        up = _rotate_xy_about_z(up0, angle)
        plate = nearest_anomaly_png(
            anomaly_plates or {},
            abs_angle,
            atol_deg=max(0.51, 0.5 * step),
        )
        bits.append(
            acquisition_camera_solids(
                setup,
                origin=pos,
                look_at=look,
                up=up,
                transmit=transmit,
                header_comment=(
                    f"orbit angle_deg={angle:g} transmit={transmit:.4g} "
                    f"(z-axis radius preserved)"
                ),
                anomaly_png=plate,
            )
        )
        if draw_frustum:
            for a, b in camera_frustum_segments(
                setup, origin=pos, look_at=look, up=up
            ):
                bits.append(
                    cylinder(
                        a,
                        b,
                        frustum_radius,
                        pigment=_camera_pigment((0.2, 0.2, 0.25), transmit),
                    )
                )
    return "".join(bits)


def coordinate_grid(z_floor: float, span: float, origin_xy=(0.0, 0.0)) -> str:
    """Subtle x–y grid on a glassy plane; faint z arrow. Visualization of axes only."""
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    n = 8
    step = float(span) / n
    r = max(span * 0.0015, 0.04)
    chunks = [
        comment_block(
            "Coordinate base: plane slightly below the mesh AABB. "
            "Grid is the simulation x–y plane (z-up). Not a measured artifact."
        ),
        plane_z(
            z_floor,
            pigment="color rgb <0.62, 0.63, 0.65>",
            finish="finish { ambient 0.05 diffuse 0.70 specular 0.35 roughness 0.04 reflection 0.12 }",
        ),
    ]
    for i in range(-n, n + 1):
        x = ox + i * step
        chunks.append(
            cylinder(
                (x, oy - span, z_floor + r),
                (x, oy + span, z_floor + r),
                r * 0.35,
                pigment="rgbt <0.75, 0.78, 0.82, 0.55>",
            )
        )
        y = oy + i * step
        chunks.append(
            cylinder(
                (ox - span, y, z_floor + r),
                (ox + span, y, z_floor + r),
                r * 0.35,
                pigment="rgbt <0.75, 0.78, 0.82, 0.55>",
            )
        )
    axis_r = r * 1.8
    chunks.append(
        comment_block("Axis hints: +x (red-gray), +y (green-gray), +z (blue-gray).")
    )
    chunks.append(
        cylinder(
            (ox, oy, z_floor + r),
            (ox + span * 0.45, oy, z_floor + r),
            axis_r,
            pigment="rgb <0.75, 0.25, 0.22>",
        )
    )
    chunks.append(
        cylinder(
            (ox, oy, z_floor + r),
            (ox, oy + span * 0.45, z_floor + r),
            axis_r,
            pigment="rgb <0.22, 0.55, 0.28>",
        )
    )
    chunks.append(
        cylinder(
            (ox, oy, z_floor),
            (ox, oy, z_floor + span * 0.35),
            axis_r * 0.7,
            pigment="rgbt <0.25, 0.35, 0.7, 0.35>",
        )
    )
    return "".join(chunks)


def scene_illumination_lights(setup: PhysicalSetup, mesh_bounds: np.ndarray) -> str:
    """POV-Ray lights for the figure only — not the catalog point source.

    Key light sits on the acquisition-camera side of the phantom.
    """
    bounds = np.asarray(mesh_bounds, dtype=float)
    lo, hi = bounds[0], bounds[1]
    center = 0.5 * (lo + hi)
    diag = float(np.linalg.norm(hi - lo))
    if diag < 1.0:
        diag = 1.0
    acq = np.asarray(setup.camera_position, dtype=float) - np.asarray(
        setup.camera_look_at, dtype=float
    )
    acq_n = acq / max(float(np.linalg.norm(acq)), 1e-9)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    key = center + 0.90 * diag * acq_n + 0.55 * diag * z_axis
    fill = center - 0.75 * diag * acq_n + 0.35 * diag * z_axis
    return (
        comment_block(
            "Scene illumination (illustration only). Independent of "
            "setups.optical.light_position_*; that marker is not a light_source. "
            "Key light: acquisition-camera side, above the phantom."
        )
        + f"light_source {{ {pov_vec(key)} color rgb <1.35, 1.28, 1.18> }}\n"
        + f"light_source {{ {pov_vec(fill)} color rgb <0.18, 0.19, 0.22> }}\n"
    )


def sky_and_horizon() -> str:
    """Sky sphere so the infinite floor meets a visible horizon (illustration only)."""
    return (
        comment_block(
            "Sky sphere (illustration). Contrasts with the z-plane so a horizon "
            "is visible; not a measured outdoor environment."
        )
        + "sky_sphere {\n"
        + "  pigment {\n"
        + "    gradient z\n"
        + "    color_map {\n"
        + "      [0.00 color rgb <0.08, 0.18, 0.42>]\n"
        + "      [0.20 color rgb <0.04, 0.10, 0.32>]\n"
        + "      [1.00 color rgb <0.01, 0.03, 0.14>]\n"
        + "    }\n"
        + "  }\n"
        + "}\n"
    )


def illustration_camera_pose(
    setup: PhysicalSetup,
    mesh_bounds: np.ndarray,
    params: IllustrationCameraParams | None = None,
    *,
    acquisition_position: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return illustration camera ``(location, look_at, fov_deg)``."""
    if params is None:
        params = IllustrationCameraParams()
    bounds = np.asarray(mesh_bounds, dtype=float)
    floor_z = float(bounds[0, 2])
    bear = np.asarray(setup.camera_look_at, dtype=float)
    cam = (
        np.asarray(setup.camera_position, dtype=float)
        if acquisition_position is None
        else np.asarray(acquisition_position, dtype=float)
    )
    acq = cam - bear
    cam_dist = float(np.linalg.norm(acq))
    if cam_dist < 1.0:
        cam_dist = 1.0
    acq_n = acq / cam_dist
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    side = np.cross(z_axis, acq_n)
    if float(np.linalg.norm(side)) < 0.15:
        side = np.array([1.0, 0.0, 0.0], dtype=float)
    side = side / np.linalg.norm(side)
    back = 0.52
    yaw0 = 0.09
    radius = float(np.hypot(back, yaw0)) * float(params.distance_scale)
    theta = float(np.arctan2(yaw0, back) + np.deg2rad(float(params.yaw_deg)))
    loc = (
        cam
        + radius * cam_dist * (np.cos(theta) * acq_n + np.sin(theta) * side)
        + 0.07 * cam_dist * z_axis
    )
    look = 0.58 * bear + 0.42 * cam
    look[2] = 0.50 * float(look[2]) + 0.50 * floor_z
    return loc, look, float(params.fov_deg)


def illustration_camera(
    setup: PhysicalSetup,
    mesh_bounds: np.ndarray,
    params: IllustrationCameraParams | None = None,
    *,
    acquisition_position: np.ndarray | None = None,
) -> str:
    """POV render camera: just behind the acquisition pinhole, looking at the bear.

    Distinct from the M8 pose. A small side/up offset keeps the camera marker
    visible in the foreground instead of sitting on the optical axis.
    """
    if params is None:
        params = IllustrationCameraParams()
    loc, look, fov_deg = illustration_camera_pose(
        setup,
        mesh_bounds,
        params,
        acquisition_position=acquisition_position,
    )
    return (
        comment_block(
            "POV-Ray render camera (illustration viewpoint). Behind the M8 "
            "acquisition camera, looking toward the bear. Not the acquisition pose.\n"
            f"yaw_deg={params.yaw_deg:g} fov_deg={fov_deg:g} "
            f"distance_scale={params.distance_scale:g}"
        )
        + "camera {\n"
        + f"  location {pov_vec(loc)}\n"
        + f"  look_at {pov_vec(look)}\n"
        + "  sky <0, 0, 1>\n"
        + f"  angle {fov_deg:g}\n"
        + "}\n"
    )


def _optical_body_scale(origin: np.ndarray, look_at: np.ndarray) -> tuple[float, float]:
    dist = float(np.linalg.norm(np.asarray(look_at, dtype=float) - np.asarray(origin, dtype=float)))
    body = max(dist * 0.055, 3.2)
    half = body * 0.45
    return body, half


def iter_drawn_optical_cameras(
    setup: PhysicalSetup,
    *,
    origin: np.ndarray,
    n_each_side: int,
    step_deg: float,
    plates: dict[float, str],
) -> list[tuple[float, np.ndarray, np.ndarray, str]]:
    """``(angle_deg, pinhole, look_at, png_name)`` for the main camera and orbit ghosts."""
    look = np.asarray(setup.camera_look_at, dtype=float)
    origin = np.asarray(origin, dtype=float).reshape(3)
    out: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    main_ang = float(setup.camera_angle_deg)
    main_png = nearest_anomaly_png(plates, main_ang)
    if main_png:
        out.append((main_ang, origin.copy(), look.copy(), main_png))
    n = int(n_each_side)
    step = float(step_deg)
    atol = max(0.51, 0.5 * step) if step > 0 else 0.51
    for k in (*range(-n, 0), *range(1, n + 1)):
        ang = k * step
        abs_ang = main_ang + ang
        pos = _rotate_xy_about_z(origin, ang)
        png = nearest_anomaly_png(plates, abs_ang, atol_deg=atol)
        if png:
            out.append((abs_ang, pos, look.copy(), png))
    return out


def _plate_apparent_score(
    origin: np.ndarray,
    look: np.ndarray,
    view_loc: np.ndarray,
    view_fwd: np.ndarray,
) -> float:
    loc = np.asarray(view_loc, dtype=float).reshape(3)
    fwd = np.asarray(view_fwd, dtype=float).reshape(3)
    fn = float(np.linalg.norm(fwd))
    if fn < 1e-9:
        return -1.0
    fwd = fwd / fn
    forward, _, _ = _basis(look - origin, np.array([0.0, 0.0, 1.0]))
    body, half = _optical_body_scale(origin, look)
    plate = origin - (half + 0.015 * body) * forward
    normal = -forward
    to_cam = loc - plate
    dist = float(np.linalg.norm(to_cam))
    if dist < 1e-6:
        return -1.0
    if float(np.dot(plate - loc, fwd)) <= 0.0:
        return -1.0
    facing = float(np.dot(to_cam / dist, normal))
    if facing <= 0.0:
        return -1.0
    return facing / (dist * dist)


def consecutive_stack_pngs(
    cameras: list[tuple[float, np.ndarray, np.ndarray, str]],
    view_loc: np.ndarray,
    view_fwd: np.ndarray,
    n: int = 3,
) -> list[str]:
    """PNGs for a stack: largest-apparent plate, then the next orbit views."""
    count = int(n)
    if count <= 0 or not cameras:
        return []
    best_i = None
    best_score = -1.0
    for i, (_ang, origin, look, _png) in enumerate(cameras):
        score = _plate_apparent_score(origin, look, view_loc, view_fwd)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i is None:
        return []
    winner_ang = cameras[best_i][0]
    uniq: dict[float, str] = {}
    for ang, _origin, _look, png in cameras:
        uniq[float(ang)] = png
    angs = sorted(uniq)
    start = min(range(len(angs)), key=lambda j: _angle_delta_deg(angs[j], winner_ang))
    take = min(count, len(angs))
    return [uniq[angs[(start + k) % len(angs)]] for k in range(take)]


def inset_plate_stack(
    png_names: list[str],
    view_loc: np.ndarray,
    view_look: np.ndarray,
    fov_deg: float,
    *,
    aspect: float = 1280.0 / 960.0,
) -> str:
    """Thin upright camera-style blocks, lower-left of the illustration frame.

    Plates stand vertical (world z up, facing the horizontal view). The stack
    fans from the front plate (bottom-left, nearer) to later plates
    (top-right, further from the viewer).
    """
    if not png_names:
        return comment_block("Inset z-score plate omitted.")
    loc = np.asarray(view_loc, dtype=float).reshape(3)
    look = np.asarray(view_look, dtype=float).reshape(3)
    z_up = np.array([0.0, 0.0, 1.0])
    view_fwd, view_right, view_up = _basis(look - loc, z_up)
    horiz = np.array([view_fwd[0], view_fwd[1], 0.0], dtype=float)
    hn = float(np.linalg.norm(horiz))
    if hn < 1e-9:
        horiz = np.array([view_right[0], view_right[1], 0.0], dtype=float)
        hn = float(np.linalg.norm(horiz))
    plate_fwd, plate_right, plate_up = _basis(horiz / max(hn, 1e-12), z_up)
    view_dist = float(np.linalg.norm(look - loc))
    depth0 = max(0.26 * view_dist, 10.0)
    half_w = np.tan(np.deg2rad(float(fov_deg)) / 2.0) * depth0
    half_h = half_w / max(float(aspect), 1e-6)
    half = 0.265 * min(half_w, half_h)
    thick = max(0.012 * half, 0.10)
    depth_step = max(0.055 * half, 0.35)
    slide = 0.09 * half
    body_p = _camera_pigment((0.18, 0.18, 0.2), 0.0)
    rim = 1.04
    bits = [
        comment_block(
            "Inset z-score plate stack (illustration). Front PNG is the optical "
            "camera whose back-plate appears largest; following plates are "
            f"consecutive orbit views ({len(png_names)} blocks). Lower-left of "
            "the frame, fanned bottom-left (near) to top-right (far). Plates "
            "are thin, world-upright camera-body blocks."
        )
    ]
    # Screen lower-left of the front plate; later plates step screen-up/right.
    # Illustration camera right is opposite this view-basis right (POV sky-up).
    screen_right = -view_right
    anchor = (
        loc
        + depth0 * view_fwd
        - 0.40 * half_w * screen_right
        - 0.60 * half_h * view_up
    )
    for i, png_name in enumerate(png_names):
        center = (
            anchor
            + i * depth_step * plate_fwd
            + i * slide * screen_right
            + i * slide * view_up
        )
        hy = half * rim
        hz = half * rim
        bits.append(
            "union {\n"
            "  box { "
            f"<0, -{hy:.6g}, -{hz:.6g}>, <{thick:.6g}, {hy:.6g}, {hz:.6g}>\n"
            f"    pigment {{ {body_p} }}\n"
            "    finish { phong 0.15 ambient 0.06 }\n"
            "  }\n"
            "  mesh2 {\n"
            "    vertex_vectors { 4,\n"
            f"      <-0.008, {-half:.6g}, {-half:.6g}>,\n"
            f"      <-0.008, {-half:.6g}, {half:.6g}>,\n"
            f"      <-0.008, {half:.6g}, {half:.6g}>,\n"
            f"      <-0.008, {half:.6g}, {-half:.6g}>\n"
            "    }\n"
            "    uv_vectors { 4, <0,0>, <0,1>, <1,1>, <1,0> }\n"
            "    face_indices { 2, <0,1,2>, <0,2,3> }\n"
            "    uv_indices { 2, <0,1,2>, <0,2,3> }\n"
            "    double_illuminate\n"
            "    texture {\n"
            "      uv_mapping\n"
            f'      pigment {{ image_map {{ png "{png_name}" once interpolate 2 }} }}\n'
            "      finish { ambient 1.0 diffuse 0.0 }\n"
            "    }\n"
            "  }\n"
            f"  matrix < {plate_fwd[0]:.6g}, {plate_right[0]:.6g}, {plate_up[0]:.6g},\n"
            f"           {plate_fwd[1]:.6g}, {plate_right[1]:.6g}, {plate_up[1]:.6g},\n"
            f"           {plate_fwd[2]:.6g}, {plate_right[2]:.6g}, {plate_up[2]:.6g},\n"
            f"           {center[0]:.6g}, {center[1]:.6g}, {center[2]:.6g} >\n"
            "  no_shadow\n"
            "}\n"
        )
    n = len(png_names)
    stack_mid = (
        anchor
        + 0.5 * (n - 1) * depth_step * plate_fwd
        + 0.5 * (n - 1) * slide * screen_right
        + 0.5 * (n - 1) * slide * view_up
    )
    gap = 0.38 * half
    head_len = 0.26 * half
    shaft_len = 0.70 * half
    r_shaft = max(0.040 * half, 0.18)
    r_head = max(0.11 * half, 0.48)
    arrow_shift = 0.22 * half * view_up + 0.28 * half * screen_right
    tail = stack_mid + (half + gap) * screen_right + arrow_shift
    cone_base = tail + shaft_len * screen_right
    tip = cone_base + head_len * screen_right
    arrow_p = "rgb <0.16, 0.16, 0.18>"
    arrow_f = "finish { phong 0.25 ambient 0.12 }"
    bits.append(
        comment_block(
            "Illustration arrow (cylinder shaft + cone tip) to the right of the "
            "plate stack, shifted slightly up and right, pointing right in the "
            "POV view."
        )
    )
    bits.append(
        cylinder(
            tail,
            cone_base,
            r_shaft,
            pigment=arrow_p,
            finish=arrow_f,
            extra="no_shadow",
        )
    )
    bits.append(
        cone(
            cone_base,
            r_head,
            tip,
            0.0,
            pigment=arrow_p,
            finish=arrow_f,
            extra="no_shadow",
        )
    )
    # Mini world triad + particle, lower-right of the frame, right of the arrow.
    axis_len = 1.45 * half
    plate_s = 0.95 * axis_len
    origin = (
        tip
        + (1.20 * half + 0.88 * plate_s) * screen_right
        - 0.04 * half * view_up
    )
    axis_r = max(0.008 * half, 0.036)
    head_frac = 0.16
    axis_f = "finish { phong 0.2 ambient 0.14 }"
    bits.append(
        comment_block(
            "Mini coordinate system (simulation x red, y green-gray, z blue-gray, "
            "z-up) with a green particle ball: localization task, lower-right of "
            "the illustration, to the right of the stack arrow. A finite xy "
            "plate at the triad origin (world z = const) takes the ball shadow "
            "and a reflection."
        )
    )

    def _mini_axis(direction: np.ndarray, pigment: str) -> None:
        d = np.asarray(direction, dtype=float).reshape(3)
        dn = float(np.linalg.norm(d))
        if dn < 1e-9:
            return
        d = d / dn
        shaft_end = origin + (1.0 - head_frac) * axis_len * d
        head_end = origin + axis_len * d
        bits.append(
            cylinder(
                origin,
                shaft_end,
                axis_r,
                pigment=pigment,
                finish=axis_f,
                extra="no_shadow",
            )
        )
        bits.append(
            cone(
                shaft_end,
                axis_r * 2.6,
                head_end,
                0.0,
                pigment=pigment,
                finish=axis_f,
                extra="no_shadow",
            )
        )

    _mini_axis(np.array([1.0, 0.0, 0.0]), "rgb <0.75, 0.25, 0.22>")
    _mini_axis(np.array([0.0, 1.0, 0.0]), "rgb <0.28, 0.62, 0.32>")
    _mini_axis(np.array([0.0, 0.0, 1.0]), "rgb <0.28, 0.42, 0.78>")
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    bits.append(
        comment_block(
            "Localization xy plate: finite piece of the simulation x–y plane "
            "(z-up) under the mini triad so the particle ball casts a shadow "
            "and a reflection."
        )
    )
    bits.append(
        "box { "
        f"<{ox - plate_s:.6g}, {oy - plate_s:.6g}, {oz - 0.045:.6g}>, "
        f"<{ox + plate_s:.6g}, {oy + plate_s:.6g}, {oz:.6g}>\n"
        "  pigment { color rgb <0.78, 0.79, 0.81> }\n"
        "  finish { ambient 0.16 diffuse 0.48 specular 0.65 roughness 0.02 "
        "reflection 0.55 }\n"
        "}\n"
    )
    bits.append(
        f"light_source {{ <{ox:.6g}, {oy:.6g}, {oz + 2.4 * axis_len:.6g}> "
        "color rgb <0.42, 0.43, 0.45> shadowless }\n"
    )
    sr_h = np.array([screen_right[0], screen_right[1], 0.0], dtype=float)
    sn = float(np.linalg.norm(sr_h))
    if sn < 1e-9:
        sr_h = np.array([1.0, 0.0, 0.0])
    else:
        sr_h = sr_h / sn
    ball = origin + 0.42 * axis_len * sr_h + 0.22 * axis_len * np.array(
        [0.0, 0.0, 1.0]
    )
    ball_r = max(0.065 * half, 0.26)
    bits.append(
        sphere(
            ball,
            ball_r,
            pigment="color rgb <0.16, 0.62, 0.28>",
            finish=(
                "finish { ambient 0.08 diffuse 0.45 phong 0.15 "
                "emission rgb <0.06, 0.16, 0.08> }"
            ),
        )
    )
    return "".join(bits)


def unique_triangle_edge_segments(mesh: trimesh.Trimesh) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Undirected triangle edges from the STL (shared edges once)."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    seen: set[tuple[int, int]] = set()
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for tri in faces:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = int(tri[i]), int(tri[j])
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            p = vertices[key[0]]
            q = vertices[key[1]]
            if float(np.linalg.norm(q - p)) < 1e-9:
                continue
            segments.append((p.copy(), q.copy()))
    return tuple(segments)


def bear_triangle_edge_cylinders(
    mesh: trimesh.Trimesh,
    *,
    radius: float,
) -> str:
    """Dark opaque cylinders along unique STL triangle edges (illustration)."""
    r = float(radius)
    if r < 0.0:
        raise ValueError(f"mesh edge radius must be >= 0, got {r}")
    if r == 0.0:
        return comment_block("Bear triangle edges omitted (radius 0).")
    bits = [
        comment_block(
            "Bear triangle edges: unique STL edges as cylinders. "
            "Opaque dark grey (not transmissive). Illustration overlay."
        )
    ]
    pigment = "rgb <0.10, 0.10, 0.11>"
    finish = "finish { ambient 0.05 diffuse 0.35 specular 0 }"
    for a, b in unique_triangle_edge_segments(mesh):
        bits.append(
            cylinder(
                a,
                b,
                r,
                pigment=pigment,
                finish=finish,
                extra="no_shadow",
            )
        )
    return "".join(bits)


def particle_marker(
    center,
    radius: float,
    *,
    catalog_radius: float,
    light_intensity: float,
    fade_distance: float,
) -> str:
    """Green inclusion marker. Optional POV ``light_source`` so the bear is lit."""
    intensity = float(light_intensity)
    if intensity < 0.0:
        raise ValueError(f"particle light intensity must be >= 0, got {intensity}")
    bits = [
        comment_block(
            "Particle position marker at setups.particles centre. Radius is a "
            f"dot for the figure (catalog radius={catalog_radius:g} mm). "
            "Sphere finish is emissive; a co-located light_source actually "
            "illuminates the transmissive bear (illustration only)."
        )
    ]
    glow = sphere(
        center,
        radius,
        pigment="color rgb <0.14, 0.82, 0.28>",
        finish=(
            "finish { ambient 0.12 diffuse 0.08 phong 0 "
            "emission rgb <0.45, 1.7, 0.55> }"
        ),
        extra="no_shadow",
    )
    if intensity == 0.0:
        bits.append(comment_block("Particle light_source omitted (intensity 0)."))
        bits.append(glow)
        return "".join(bits)
    r, g, b = 0.22 * intensity, 1.85 * intensity, 0.38 * intensity
    fd = float(fade_distance)
    bits.append(
        comment_block(
            f"Particle illumination light_source intensity={intensity:g} "
            f"fade_distance={fd:g} (local green fill inside the phantom)."
        )
    )
    bits.append(
        "light_source {\n"
        f"  {pov_vec(center)}\n"
        f"  color rgb <{r:.4g}, {g:.4g}, {b:.4g}>\n"
        f"  fade_distance {fd:.6g}\n"
        "  fade_power 2\n"
        "}\n"
    )
    bits.append(glow)
    return "".join(bits)


def catalog_light_cone(
    light_position: np.ndarray,
    mesh_bounds: np.ndarray,
    *,
    length_frac: float = 0.5,
    n_slices: int = 10,
) -> str:
    """Transmissive cone from the catalog light toward the mesh AABB.

    Axis aims at the AABB centre. Half-angle is the smallest cone from the
    light that contains every AABB corner. Length is at most ``length_frac``
    of the light-to-centre distance. Transmit increases away from the light.
    Illustration only — not a POV-Ray ``light_source`` and not solver geometry.
    """
    light = np.asarray(light_position, dtype=float).reshape(3)
    bbox = np.asarray(mesh_bounds, dtype=float)
    if bbox.shape != (2, 3):
        raise ValueError(f"mesh_bounds must be [2, 3], got {bbox.shape}")
    center = bbox.mean(axis=0)
    axis = center - light
    dist = float(np.linalg.norm(axis))
    if float(length_frac) <= 0.0:
        return comment_block("Catalog light cone omitted (length_frac <= 0).")
    if dist < 1e-9:
        return comment_block("Catalog light cone omitted (light at AABB centre).")
    axis_n = axis / dist
    corners = np.array(
        [
            [bbox[i, 0], bbox[j, 1], bbox[k, 2]]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ],
        dtype=float,
    )
    half = 0.0
    for corner in corners:
        vec = corner - light
        nrm = float(np.linalg.norm(vec))
        if nrm < 1e-12:
            continue
        cang = float(np.clip(np.dot(axis_n, vec / nrm), -1.0, 1.0))
        half = max(half, float(np.arccos(cang)))
    if half < 1e-6:
        return comment_block("Catalog light cone omitted (degenerate AABB).")
    half = min(half, float(np.deg2rad(80.0)))
    length = float(length_frac) * dist
    tan_h = float(np.tan(half))
    chunks = [
        comment_block(
            "Catalog illumination cone: apex at light_position, axis toward "
            "mesh AABB centre, half-angle covers the AABB, length "
            f"{length_frac:g} of light-to-centre. Intensity fades to 0 at the far end."
        )
    ]
    for i in range(int(n_slices)):
        t0 = i / n_slices
        t1 = (i + 1) / n_slices
        p0 = light + t0 * length * axis_n
        p1 = light + t1 * length * axis_n
        r0 = t0 * length * tan_h
        r1 = t1 * length * tan_h
        intensity = max(1.0 - t1, 0.0)
        transmit = 1.0 - 0.72 * intensity
        em = 0.28 * intensity
        chunks.append(
            cone(
                p0,
                max(r0, 1e-4),
                p1,
                max(r1, 1e-4),
                pigment=f"color rgbt <1.0, 0.90, 0.35, {transmit:.3g}>",
                finish=(
                    "finish { ambient 0.85 diffuse 0.05 "
                    f"emission rgb <{em:.3g}, {0.80 * em:.3g}, {0.16 * em:.3g}> }}"
                ),
            )
        )
    return "".join(chunks)


def build_pov_scene(
    setup: PhysicalSetup,
    *,
    bear_inc_name: str,
    object_name: str = "BearMesh",
    illustration_camera_params: IllustrationCameraParams | None = None,
    particle_radius_mm: float | None = None,
    light_cone_length_frac: float | None = None,
    optical_camera_distance_scale: float | None = None,
    catalog_light_distance_scale: float | None = None,
    mesh_edge_radius_mm: float | None = None,
    orbit_cameras: int | None = None,
    orbit_step_deg: float | None = None,
    camera_rays: str | None = None,
    particle_light: float | None = None,
    anomaly_plates: dict[float, str] | None = None,
    orbit_fade: bool | None = None,
    inset_plate: bool | None = None,
    inset_stack: int | None = None,
) -> str:
    mesh = trimesh.load(setup.stl_path, force="mesh")
    bounds = np.asarray(mesh.bounds, dtype=float)
    lo, hi = bounds[0], bounds[1]
    span = 0.5 * float(np.max(hi - lo))
    z_floor = float(lo[2]) - 0.04 * float(hi[2] - lo[2] + 1e-6)
    origin_xy = (float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1])))
    ray_radius = max(span * 0.004, 0.08)
    optical_ray_radius = ray_radius * 0.55

    cam_scale = (
        1.0
        if optical_camera_distance_scale is None
        else float(optical_camera_distance_scale)
    )
    light_scale = (
        1.0
        if catalog_light_distance_scale is None
        else float(catalog_light_distance_scale)
    )
    acq_pos = illustrated_acquisition_position(setup, distance_scale=cam_scale)
    light_pos = illustrated_catalog_light_position(
        setup, bounds, distance_scale=light_scale
    )

    header = [
        "#version 3.7;",
        "global_settings { assumed_gamma 1.0 ambient_light rgb <0.28, 0.28, 0.30> }",
        "",
        comment_block(
            f"M8 physical setup  sequence_id={setup.sequence_id}\n"
            f"manifest={display_path(setup.manifest_path)}\n"
            f"stl={display_path(setup.stl_path)}\n"
            f"camera_angle_deg={setup.camera_angle_deg:g} (acquisition object)\n"
            f"illustration optical_camera_distance_scale={cam_scale:g} "
            f"catalog_light_distance_scale={light_scale:g} "
            f"orbit_cameras={4 if orbit_cameras is None else int(orbit_cameras)} "
            f"orbit_step_deg="
            f"{20 if orbit_step_deg is None else float(orbit_step_deg):g} "
            f"camera_rays="
            f"{'all' if camera_rays is None else str(camera_rays).strip().lower()}\n"
            "World axes = simulation millimetres, z-up."
        ),
    ]
    if setup.warnings:
        header.append(comment_block("Warnings:\n" + "\n".join(setup.warnings)))
    if setup.illumination_rays_are_fallback:
        header.append(
            comment_block(
                "FALLBACK: source cylinders from make_source_ray_bundle "
                "(point_uniform) to first hits; interior cylinders from "
                "refract_ray_bundle + in_object_segments_from_rays. "
                "n_rays is illustration-downsampled."
            )
        )

    edge_r = (
        max(span * 0.0035, 0.05)
        if mesh_edge_radius_mm is None
        else float(mesh_edge_radius_mm)
    )
    bear = (
        comment_block(
            "Bear mesh: converted STL. Neutral grey, ~72% transmit, no gloss."
        )
        + f'#include "{bear_inc_name}"\n'
        + "object {\n"
        + f"  {object_name}\n"
        + "  pigment { color rgbt <0.68, 0.69, 0.71, 0.72> }\n"
        + "  finish { ambient 0.06 diffuse 0.55 specular 0 roughness 0.5 reflection 0 }\n"
        + "  interior { ior 1.02 }\n"
        + "}\n"
        + bear_triangle_edge_cylinders(mesh, radius=edge_r)
    )

    marker_r = max(span * 0.022, 0.35)
    particle_r = (
        float(particle_radius_mm)
        if particle_radius_mm is not None
        else max(span * 0.028, 0.42)
    )
    if particle_r <= 0.0:
        raise ValueError(f"particle_radius_mm must be > 0, got {particle_r}")
    particle_intensity = 1.0 if particle_light is None else float(particle_light)
    particle = particle_marker(
        setup.particle_center,
        particle_r,
        catalog_radius=setup.particle_radius,
        light_intensity=particle_intensity,
        fade_distance=max(span * 0.65, particle_r * 8.0),
    )

    light = comment_block(
        "Catalog point-light position marker (setups.optical.light_position_*). "
        "Not a POV-Ray light_source; size is a dot, not a physical radius."
    ) + sphere(
        light_pos,
        marker_r * 1.3,
        pigment="color rgb <1.0, 0.92, 0.35>",
        finish="finish { ambient 0.95 diffuse 0.1 emission rgb <0.45, 0.38, 0.08> }",
        extra="no_shadow",
    )

    ray_bits = [
        comment_block(
            "Illumination (light → entry) and refracted in-object chords. "
            "High ambient so catalog rays read as glowing. FALLBACK when rebuilt."
        )
    ]
    pigment = (
        "rgbt <1.0, 0.88, 0.25, 0.08>"
        if setup.illumination_rays_are_fallback
        else "rgbt <1.0, 0.82, 0.18, 0.06>"
    )
    ray_finish = (
        "finish { ambient 0.55 diffuse 0.25 emission rgb <0.28, 0.22, 0.05> }"
    )
    for a, b in setup.illumination_rays:
        ray_bits.append(
            cylinder(a, b, optical_ray_radius, pigment=pigment, finish=ray_finish)
        )
    for a, b in setup.refracted_rays:
        ray_bits.append(
            cylinder(a, b, optical_ray_radius, pigment=pigment, finish=ray_finish)
        )

    plates = anomaly_plates or {}
    main_plate = nearest_anomaly_png(plates, float(setup.camera_angle_deg))
    cam_obj = acquisition_camera_solids(
        setup, origin=acq_pos, anomaly_png=main_plate
    )
    rays_mode = "all" if camera_rays is None else str(camera_rays).strip().lower()
    if rays_mode not in {"all", "single", "none"}:
        raise ValueError(
            "camera_rays must be 'all', 'single', or 'none', "
            f"got {camera_rays!r}"
        )
    draw_main_frustum = rays_mode in {"all", "single"}
    draw_orbit_frustum = rays_mode == "all"
    if draw_main_frustum:
        frustum = comment_block(
            "Camera frustum edges from illustrated pinhole + catalog fov_deg "
            "(same look-at; distance may be illustration-scaled)."
        )
        for a, b in camera_frustum_segments(setup, origin=acq_pos):
            frustum += cylinder(
                a, b, ray_radius * 0.4, pigment="rgbt <0.2, 0.2, 0.25, 0.35>"
            )
    else:
        frustum = comment_block("Camera frustum omitted (camera_rays=none).")
    n_orbit = 4 if orbit_cameras is None else int(orbit_cameras)
    step_orbit = 20.0 if orbit_step_deg is None else float(orbit_step_deg)
    fade_orbit = True if orbit_fade is None else bool(orbit_fade)
    orbit = tomography_orbit_cameras(
        setup,
        origin=acq_pos,
        n_each_side=n_orbit,
        step_deg=step_orbit,
        frustum_radius=ray_radius * 0.4,
        draw_frustum=draw_orbit_frustum,
        anomaly_plates=plates,
        fade=fade_orbit,
    )

    cam_params = (
        IllustrationCameraParams()
        if illustration_camera_params is None
        else illustration_camera_params
    )
    view_loc, view_look, view_fov = illustration_camera_pose(
        setup,
        bounds,
        cam_params,
        acquisition_position=acq_pos,
    )
    inset = comment_block("Inset z-score plate omitted.")
    if inset_plate is not False and plates:
        drawn = iter_drawn_optical_cameras(
            setup,
            origin=acq_pos,
            n_each_side=n_orbit,
            step_deg=step_orbit,
            plates=plates,
        )
        pngs = consecutive_stack_pngs(
            drawn,
            view_loc,
            view_look - view_loc,
            n=3 if inset_stack is None else int(inset_stack),
        )
        if pngs:
            inset = inset_plate_stack(
                pngs,
                view_loc,
                view_look,
                view_fov,
            )

    bg = sky_and_horizon()

    return "\n".join(
        [
            *header,
            bg,
            illustration_camera(
                setup,
                bounds,
                illustration_camera_params,
                acquisition_position=acq_pos,
            ),
            scene_illumination_lights(setup, bounds),
            coordinate_grid(z_floor, span, origin_xy),
            bear,
            particle,
            light,
            catalog_light_cone(
                light_pos,
                bounds,
                length_frac=(
                    0.5
                    if light_cone_length_frac is None
                    else float(light_cone_length_frac)
                ),
            ),
            "".join(ray_bits),
            cam_obj,
            frustum,
            orbit,
            inset,
        ]
    )
