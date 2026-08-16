"""Schematic POV-Ray 3.7 scene for M8 network inference."""

from __future__ import annotations

import numpy as np

from .network_activations import CHANNEL_DEPTHS, NetworkActivationBundle
from .pov_primitives import box, comment_block, cone, cylinder, plane_z, pov_vec, sphere
from .pov_scene import _basis, _camera_pigment, sky_and_horizon

PIPELINE_PIVOT = np.array([48.0, 0.0, 0.0], dtype=float)
FACE_HALF = 6.4 * 1.5
DEPTH_HALF = 6.4
CNN_Z_CENTER = FACE_HALF + 1.6
# Further left in the frame (larger world y after the pipeline rotate).
DEFAULT_INPUT_STACK_CENTER = np.array([48.0, 58.0, CNN_Z_CENTER], dtype=float)
# Fourier embedding slab vs the 2×2 pane: extra front gap, screen-right, lift from bottom.
DEFAULT_FOURIER_EMBED_OFFSET = (0.28, 2.6, 0.0)
DEFAULT_FOURIER_GROUP_OFFSET = (0.0, 0.0, 0.0)
_PANE_HALF_THICK = 0.16
_EMBED_SLAB_RADIUS = 1.15
_FLOOR_Z = -0.4
_BEAR_FLOOR_GAP = 0.15
_GAP_SLAB_LENGTH = 22.0
_GAP_SLAB_RADIUS = 1.15
FOURIER_ARROW_PIGMENT = "rgb <0.14, 0.38, 0.62>"
POOLING_ARROW_PIGMENT = "rgb <0.52, 0.08, 0.10>"
FOURIER_TO_BALL_PIGMENT = "rgb <0.16, 0.34, 0.82>"
_LOCALIZATION_BEAR_SPAN = 22.0


def illustration_camera_pose(
    *,
    yaw_deg: float,
    distance_scale: float,
    look: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Illustration camera location and look-at (z-up)."""
    look_at = (
        np.array([48.0, 0.0, 7.0], dtype=float)
        if look is None
        else np.asarray(look, dtype=float).reshape(3)
    )
    dist = 95.0 * float(distance_scale)
    yaw = np.deg2rad(float(yaw_deg))
    loc = look_at + np.array(
        [dist * np.sin(yaw), -dist * np.cos(yaw), 28.0 * float(distance_scale)]
    )
    return loc, look_at


def _rot_z(vec: np.ndarray, yaw_deg: float) -> np.ndarray:
    v = np.asarray(vec, dtype=float).reshape(3)
    a = np.deg2rad(float(yaw_deg))
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=float)


def _horiz_toward_camera(view_loc: np.ndarray, view_look: np.ndarray) -> np.ndarray:
    """Unit world vector on the floor plane, from the look-at toward the camera."""
    loc = np.asarray(view_loc, dtype=float).reshape(3)
    look = np.asarray(view_look, dtype=float).reshape(3)
    horiz = np.array([loc[0] - look[0], loc[1] - look[1], 0.0], dtype=float)
    n = float(np.linalg.norm(horiz))
    if n < 1e-9:
        return np.array([0.0, -1.0, 0.0], dtype=float)
    return horiz / n


def _bear_marker_pipeline(
    *,
    origin_xy: np.ndarray,
    mesh_centroid: np.ndarray,
    mesh_extent: float,
    mesh_zmin: float | None,
    xyz: np.ndarray,
    target_span: float,
) -> np.ndarray:
    """Pipeline coordinates of a particle marker on a scaled floor-sitting bear."""
    c = np.asarray(mesh_centroid, dtype=float).reshape(3)
    origin_xy = np.asarray(origin_xy, dtype=float).reshape(3)
    extent = max(float(mesh_extent), 1e-6)
    scale = float(target_span) / extent
    cz = float(c[2])
    zmin = cz if mesh_zmin is None else float(mesh_zmin)
    oz = _FLOOR_Z + _BEAR_FLOOR_GAP - scale * (zmin - cz)
    origin = np.array([float(origin_xy[0]), float(origin_xy[1]), oz], dtype=float)
    return origin + scale * (np.asarray(xyz, dtype=float).reshape(3) - c)


def pipeline_to_world(point: np.ndarray, *, pivot: np.ndarray | None = None) -> np.ndarray:
    """Map schematic pipeline coordinates through the scene's −90° z rotation."""
    pvt = PIPELINE_PIVOT if pivot is None else np.asarray(pivot, dtype=float).reshape(3)
    r = np.asarray(point, dtype=float).reshape(3) - pvt
    return pvt + np.array([r[1], -r[0], r[2]], dtype=float)


def _header() -> str:
    return (
        comment_block(
            "M8 network-inference illustration (schematic, not catalog millimetres). "
            "Captions are drawn on the PNG after POV-Ray; do not add POV text."
        )
        + "#version 3.7;\n"
        + "global_settings { assumed_gamma 1.0 }\n"
        + "background { color rgb <0.04, 0.06, 0.10> }\n"
        + sky_and_horizon()
    )


def _camera(*, yaw_deg: float, fov_deg: float, distance_scale: float) -> str:
    loc, look = illustration_camera_pose(yaw_deg=yaw_deg, distance_scale=distance_scale)
    return (
        comment_block("Illustration camera (pipeline overview, z-up).")
        + "camera {\n"
        + f"  location {pov_vec(loc)}\n"
        + f"  look_at {pov_vec(look)}\n"
        + "  sky <0, 0, 1>\n"
        + f"  angle {float(fov_deg):.4g}\n"
        + "  right x * image_width / image_height\n"
        + "}\n"
        + f"light_source {{ {pov_vec(loc + np.array([8.0, -12.0, 22.0]))} "
        "color rgb <1.25, 1.20, 1.12> }\n"
        + "light_source { <-20, -40, 40> color rgb <0.22, 0.23, 0.26> }\n"
    )


def _floor() -> str:
    return plane_z(
        _FLOOR_Z,
        pigment="color rgb <0.62, 0.63, 0.65>",
        finish=(
            "finish { ambient 0.05 diffuse 0.70 specular 0.35 "
            "roughness 0.04 reflection 0.12 }"
        ),
    )


def _quad_neg_y(
    png: str,
    center: np.ndarray,
    half_x: float,
    half_z: float,
) -> str:
    cx, cy, cz = (float(v) for v in np.asarray(center, dtype=float).reshape(3))
    return (
        "mesh2 {\n"
        "  vertex_vectors { 4,\n"
        f"    <{cx - half_x:.6g}, {cy:.6g}, {cz - half_z:.6g}>,\n"
        f"    <{cx - half_x:.6g}, {cy:.6g}, {cz + half_z:.6g}>,\n"
        f"    <{cx + half_x:.6g}, {cy:.6g}, {cz + half_z:.6g}>,\n"
        f"    <{cx + half_x:.6g}, {cy:.6g}, {cz - half_z:.6g}>\n"
        "  }\n"
        "  uv_vectors { 4, <0,0>, <0,1>, <1,1>, <1,0> }\n"
        "  face_indices { 2, <0,1,2>, <0,2,3> }\n"
        "  uv_indices { 2, <0,1,2>, <0,2,3> }\n"
        "  double_illuminate\n"
        "  texture {\n"
        "    uv_mapping\n"
        f'    pigment {{ image_map {{ png "{png}" once interpolate 2 }} }}\n'
        "    finish { ambient 0.38 diffuse 0.62 phong 0.08 }\n"
        "  }\n"
        "}\n"
    )


def _face_quad(
    png: str,
    v0,
    v1,
    v2,
    v3,
    *,
    interpolate: int | None = 2,
    u0: float = 0.0,
    u1: float = 1.0,
    v_lo: float = 0.0,
    v_hi: float = 1.0,
) -> str:
    """Textured rectangle; ``double_illuminate`` so both sides (and reflections) read."""
    pts = [np.asarray(v, dtype=float).reshape(3) for v in (v0, v1, v2, v3)]
    lines = ",\n    ".join(pov_vec(p) for p in pts)
    if interpolate is None:
        imap = f'png "{png}" once'
    else:
        imap = f'png "{png}" once interpolate {int(interpolate)}'
    return (
        "mesh2 {\n"
        f"  vertex_vectors {{ 4,\n    {lines}\n  }}\n"
        f"  uv_vectors {{ 4, <{u0:g},{v_lo:g}>, <{u0:g},{v_hi:g}>, "
        f"<{u1:g},{v_hi:g}>, <{u1:g},{v_lo:g}> }}\n"
        "  face_indices { 2, <0,1,2>, <0,2,3> }\n"
        "  uv_indices { 2, <0,1,2>, <0,2,3> }\n"
        "  double_illuminate\n"
        "  texture {\n"
        "    uv_mapping\n"
        f"    pigment {{ image_map {{ {imap} }} }}\n"
        "    finish { ambient 0.38 diffuse 0.62 phong 0.08 }\n"
        "  }\n"
        "}\n"
    )


def _cuboid_activation_skins(
    lo: np.ndarray,
    hi: np.ndarray,
    faces: dict[str, str],
    *,
    eps: float = 0.045,
) -> str:
    """Actual ``[C,H,W]`` boundary slices on all six faces (shared colormap)."""
    x0, y0, z0 = (float(v) for v in np.asarray(lo, dtype=float).reshape(3))
    x1, y1, z1 = (float(v) for v in np.asarray(hi, dtype=float).reshape(3))
    x0e, x1e = x0 - eps, x1 + eps
    y0e, y1e = y0 - eps, y1 + eps
    z0e, z1e = z0 - eps, z1 + eps
    bits = [
        comment_block(
            "Activation skins: six boundary slices of the CNN tensor "
            "(x=channel, y=width, z=height), per-channel colormap."
        )
    ]
    bits.append(
        _face_quad(
            faces["xm"],
            (x0e, y0, z0),
            (x0e, y0, z1),
            (x0e, y1, z1),
            (x0e, y1, z0),
        )
    )
    bits.append(
        _face_quad(
            faces["xp"],
            (x1e, y0, z0),
            (x1e, y0, z1),
            (x1e, y1, z1),
            (x1e, y1, z0),
        )
    )
    bits.append(
        _face_quad(
            faces["ym"],
            (x0, y0e, z0),
            (x0, y0e, z1),
            (x1, y0e, z1),
            (x1, y0e, z0),
        )
    )
    bits.append(
        _face_quad(
            faces["yp"],
            (x0, y1e, z0),
            (x0, y1e, z1),
            (x1, y1e, z1),
            (x1, y1e, z0),
        )
    )
    bits.append(
        _face_quad(
            faces["zm"],
            (x0, y0, z0e),
            (x0, y1, z0e),
            (x1, y1, z0e),
            (x1, y0, z0e),
        )
    )
    bits.append(
        _face_quad(
            faces["zp"],
            (x0, y0, z1e),
            (x0, y1, z1e),
            (x1, y1, z1e),
            (x1, y0, z1e),
        )
    )
    return "".join(bits)


def _arrow(
    tail,
    tip,
    *,
    r_shaft: float = 0.22,
    r_head: float = 0.48,
    pigment: str = FOURIER_ARROW_PIGMENT,
    finish: str = "finish { phong 0.3 ambient 0.22 diffuse 0.45 }",
    keep_head: bool = False,
) -> str:
    tail = np.asarray(tail, dtype=float)
    tip = np.asarray(tip, dtype=float)
    d = tip - tail
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return ""
    d = d / n
    head_len = min(2.15, 0.45 * n)
    if keep_head:
        head_len = min(2.15, max(0.35, n - 0.35))
    cone_base = tip - head_len * d
    return cylinder(
        tail, cone_base, r_shaft, pigment=pigment, finish=finish, extra=""
    ) + cone(
        cone_base,
        r_head * 2.5,
        tip,
        0.0,
        pigment=pigment,
        finish=finish,
        extra="",
    )


def _inset_arrow(
    tail,
    tip,
    *,
    tail_frac: float = 0.20,
    tip_frac: float = 0.38,
    keep_head: bool = False,
    pigment: str = FOURIER_ARROW_PIGMENT,
) -> str:
    tail = np.asarray(tail, dtype=float).reshape(3)
    tip = np.asarray(tip, dtype=float).reshape(3)
    delta = tip - tail
    if float(np.linalg.norm(delta)) < 1e-9:
        return ""
    return _arrow(
        tail + tail_frac * delta,
        tip - tip_frac * delta,
        keep_head=keep_head,
        pigment=pigment,
    )


def _plate_image_mesh(png: str, *, x: float, half: float, outward_plus_x: bool) -> str:
    h = float(half)
    if outward_plus_x:
        verts = (
            f"      <{x:.6g}, {h:.6g}, {-h:.6g}>,\n"
            f"      <{x:.6g}, {h:.6g}, {h:.6g}>,\n"
            f"      <{x:.6g}, {-h:.6g}, {h:.6g}>,\n"
            f"      <{x:.6g}, {-h:.6g}, {-h:.6g}>\n"
        )
    else:
        verts = (
            f"      <{x:.6g}, {-h:.6g}, {-h:.6g}>,\n"
            f"      <{x:.6g}, {-h:.6g}, {h:.6g}>,\n"
            f"      <{x:.6g}, {h:.6g}, {h:.6g}>,\n"
            f"      <{x:.6g}, {h:.6g}, {-h:.6g}>\n"
        )
    return (
        "  mesh2 {\n"
        f"    vertex_vectors {{ 4,\n{verts}    }}\n"
        "    uv_vectors { 4, <0,0>, <0,1>, <1,1>, <1,0> }\n"
        "    face_indices { 2, <0,1,2>, <0,2,3> }\n"
        "    uv_indices { 2, <0,1,2>, <0,2,3> }\n"
        "    double_illuminate\n"
        "    texture {\n"
        "      uv_mapping\n"
        f'      pigment {{ image_map {{ png "{png}" once interpolate 2 }} }}\n'
        "      finish { ambient 0.38 diffuse 0.62 phong 0.08 }\n"
        "    }\n"
        "  }\n"
    )


def _plate_axes(
    view_loc: np.ndarray,
    view_look: np.ndarray,
    *,
    extra_yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-up plate basis; ``extra_yaw_deg`` from camera-facing."""
    loc = np.asarray(view_loc, dtype=float).reshape(3)
    look = np.asarray(view_look, dtype=float).reshape(3)
    z_up = np.array([0.0, 0.0, 1.0])
    view_fwd, _view_right, _view_up = _basis(look - loc, z_up)
    horiz = np.array([view_fwd[0], view_fwd[1], 0.0], dtype=float)
    hn = float(np.linalg.norm(horiz))
    if hn < 1e-9:
        horiz = np.array([1.0, 0.0, 0.0], dtype=float)
        hn = 1.0
    plate_fwd, plate_right, plate_up = _basis(horiz / hn, z_up)
    yaw = np.deg2rad(float(extra_yaw_deg))
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    plate_fwd = np.array(
        [c * plate_fwd[0] - s * plate_fwd[1], s * plate_fwd[0] + c * plate_fwd[1], plate_fwd[2]],
        dtype=float,
    )
    plate_right = np.array(
        [
            c * plate_right[0] - s * plate_right[1],
            s * plate_right[0] + c * plate_right[1],
            plate_right[2],
        ],
        dtype=float,
    )
    return plate_fwd, plate_right, plate_up


def _input_stack(
    pngs: list[str],
    *,
    center: np.ndarray,
    view_loc: np.ndarray,
    view_look: np.ndarray,
    half: float,
    comment: str | None = None,
    extra_yaw_deg: float = 90.0,
) -> str:
    """Single-view plate: world-up, yawed 90°, z-score texture on both faces."""
    if not pngs:
        return comment_block("Input z-score plate omitted.")
    plate_fwd, plate_right, plate_up = _plate_axes(
        view_loc, view_look, extra_yaw_deg=extra_yaw_deg
    )
    thick = 0.32
    depth_step = max(0.055 * float(half), 0.35)
    slide = 0.09 * float(half)
    body_p = _camera_pigment((0.18, 0.18, 0.2), 0.0)
    rim = 1.04
    n = len(pngs)
    mid = 0.5 * (n - 1)
    stack_c = np.asarray(center, dtype=float).reshape(3)
    bits = [
        comment_block(
            comment
            if comment is not None
            else (
                "Input z-score plate (single-view): thicker camera-body block, "
                "world-up, yawed 90° toward the pipeline, image on both faces. "
                f"Centre {pov_vec(stack_c)}."
            )
        )
    ]
    ht = 0.5 * thick
    for i, png_name in enumerate(pngs):
        k = float(i) - mid
        plate_c = stack_c + k * depth_step * plate_fwd + k * slide * plate_right
        hy = float(half) * rim
        hz = float(half) * rim
        bits.append("union {\n")
        bits.append(
            "  box { "
            f"<-{ht:.6g}, -{hy:.6g}, -{hz:.6g}>, <{ht:.6g}, {hy:.6g}, {hz:.6g}>\n"
            f"    pigment {{ {body_p} }}\n"
            "    finish { phong 0.15 ambient 0.06 }\n"
            "  }\n"
        )
        bits.append(
            _plate_image_mesh(
                png_name, x=-ht - 0.008, half=float(half), outward_plus_x=False
            )
        )
        bits.append(
            _plate_image_mesh(
                png_name, x=ht + 0.008, half=float(half), outward_plus_x=True
            )
        )
        bits.append(
            f"  matrix < {plate_fwd[0]:.6g}, {plate_right[0]:.6g}, {plate_up[0]:.6g},\n"
            f"           {plate_fwd[1]:.6g}, {plate_right[1]:.6g}, {plate_up[1]:.6g},\n"
            f"           {plate_fwd[2]:.6g}, {plate_right[2]:.6g}, {plate_up[2]:.6g},\n"
            f"           {plate_c[0]:.6g}, {plate_c[1]:.6g}, {plate_c[2]:.6g} >\n"
            "}\n"
        )
    return "".join(bits)


def _oriented_pooling_slab(
    png: str,
    *,
    origin: np.ndarray,
    long_dir: np.ndarray,
    toward_camera: np.ndarray,
    up_dir: np.ndarray,
    length: float,
    radius: float,
    comment: str,
) -> str:
    """World-space pooling bar: local x along ``long_dir``, +y toward the camera."""
    long_u = np.asarray(long_dir, dtype=float).reshape(3)
    cam_u = np.asarray(toward_camera, dtype=float).reshape(3)
    up_u = np.asarray(up_dir, dtype=float).reshape(3)
    long_u = long_u / float(np.linalg.norm(long_u))
    cam_u = cam_u / float(np.linalg.norm(cam_u))
    up_u = up_u / float(np.linalg.norm(up_u))
    ox, oy, oz = (float(v) for v in np.asarray(origin, dtype=float).reshape(3))
    bits = ["union {\n"]
    bits.append(
        _gap_slab(
            png,
            origin=np.zeros(3),
            length=length,
            radius=radius,
            comment=comment,
        )
    )
    bits.append(
        f"  matrix < {long_u[0]:.6g}, {cam_u[0]:.6g}, {up_u[0]:.6g},\n"
        f"           {long_u[1]:.6g}, {cam_u[1]:.6g}, {up_u[1]:.6g},\n"
        f"           {long_u[2]:.6g}, {cam_u[2]:.6g}, {up_u[2]:.6g},\n"
        f"           {ox:.6g}, {oy:.6g}, {oz:.6g} >\n"
        "}\n"
    )
    return "".join(bits)


def _cnn_volumes(
    face_sets: list[dict[str, str]],
    *,
    start_x: float,
    half: float,
    depth_half: float,
    zc: float,
    comment: str | None = None,
) -> tuple[str, float]:
    bits = [
        comment_block(
            comment
            if comment is not None
            else (
                "CNN feature volumes: Encode 16→32→64 channels, spatial 128×128 "
                "(same footprint, increasing channel depth). Each face is a "
                "boundary slice of the real [C,H,W] tensor; colours use a "
                "per-channel scale (same as the inspection notebook)."
            )
        )
    ]
    x = float(start_x)
    for faces, nch in zip(face_sets, CHANNEL_DEPTHS, strict=True):
        thick = float(depth_half) * (float(nch) / 64.0) * 1.15
        lo = np.array([x, -half, zc - half])
        hi = np.array([x + thick, half, zc + half])
        bits.append(box(lo, hi, pigment="color rgb <0.16, 0.17, 0.19>"))
        bits.append(_cuboid_activation_skins(lo, hi, faces))
        x = x + thick + 3.8
    return "".join(bits), x


def _cnn_end_x(*, start_x: float, depth_half: float) -> float:
    x = float(start_x)
    for nch in CHANNEL_DEPTHS:
        x += float(depth_half) * (float(nch) / 64.0) * 1.15 + 3.8
    return x


def _pipeline_union(body: str, *, extra_world: np.ndarray | None = None) -> str:
    """Pipeline solids with the scene −90° z rotation; optional world shift."""
    pivot_x, pivot_y = 48.0, 0.0
    bits = [
        "union {\n",
        body,
        f"  translate <-{pivot_x:.6g}, {-pivot_y:.6g}, 0>\n",
        "  rotate <0, 0, -90>\n",
        f"  translate <{pivot_x:.6g}, {pivot_y:.6g}, 0>\n",
    ]
    if extra_world is not None:
        w = np.asarray(extra_world, dtype=float).reshape(3)
        bits.append(f"  translate <{w[0]:.6g}, {w[1]:.6g}, {w[2]:.6g}>\n")
    bits.append("}\n")
    return "".join(bits)


def _strip_bin_uv(index: int, n: int = 64) -> tuple[float, float]:
    """Open interval inside one column of a 1-D strip (avoid u=0/1 ``once`` edges)."""
    n = max(int(n), 1)
    i = min(max(int(index), 0), n - 1)
    return (i + 0.2) / n, (i + 0.8) / n


def _gap_slab(
    png: str,
    *,
    origin: np.ndarray,
    length: float = 22.0,
    radius: float = 1.15,
    comment: str | None = None,
    n_bins: int = 64,
) -> str:
    """Long pooling bar: 64 colours along x on all six faces (reflections)."""
    ox, oy, oz = (float(v) for v in np.asarray(origin, dtype=float).reshape(3))
    L = float(length)
    r = float(radius)
    n_bins = max(int(n_bins), 1)
    u_lo, u_hi = 0.5 / n_bins, 1.0 - 0.5 / n_bins
    u0a, u0b = _strip_bin_uv(0, n_bins)
    u1a, u1b = _strip_bin_uv(n_bins - 1, n_bins)
    x0, x1 = ox, ox + L
    bits = [
        comment_block(
            comment
            if comment is not None
            else (
                "Average pooling: GAP [64] as turbo bands along a long slab "
                "(same min/max scale as the inspection strip)."
            )
        )
    ]
    bits.append(
        box(
            np.array([x0, oy - r, oz - r]),
            np.array([x1, oy + r, oz + r]),
            pigment="color rgb <0.14, 0.15, 0.17>",
        )
    )
    e = 0.02
    bits.append(
        _face_quad(
            png,
            (x0, oy + r + e, oz - r),
            (x0, oy + r + e, oz + r),
            (x1, oy + r + e, oz + r),
            (x1, oy + r + e, oz - r),
            interpolate=None,
            u0=u_lo,
            u1=u_hi,
        )
    )
    bits.append(
        _face_quad(
            png,
            (x0, oy - r - e, oz + r),
            (x0, oy - r - e, oz - r),
            (x1, oy - r - e, oz - r),
            (x1, oy - r - e, oz + r),
            interpolate=None,
            u0=u_lo,
            u1=u_hi,
        )
    )
    bits.append(
        _face_quad(
            png,
            (x0, oy - r, oz + r + e),
            (x0, oy + r, oz + r + e),
            (x1, oy + r, oz + r + e),
            (x1, oy - r, oz + r + e),
            interpolate=None,
            u0=u_lo,
            u1=u_hi,
        )
    )
    bits.append(
        _face_quad(
            png,
            (x0, oy + r, oz - r - e),
            (x0, oy - r, oz - r - e),
            (x1, oy - r, oz - r - e),
            (x1, oy + r, oz - r - e),
            interpolate=None,
            u0=u_lo,
            u1=u_hi,
        )
    )
    bits.append(
        _face_quad(
            png,
            (x0 - e, oy - r, oz - r),
            (x0 - e, oy - r, oz + r),
            (x0 - e, oy + r, oz + r),
            (x0 - e, oy + r, oz - r),
            interpolate=None,
            u0=u0a,
            u1=u0b,
        )
    )
    bits.append(
        _face_quad(
            png,
            (x1 + e, oy - r, oz - r),
            (x1 + e, oy + r, oz - r),
            (x1 + e, oy + r, oz + r),
            (x1 + e, oy - r, oz + r),
            interpolate=None,
            u0=u1a,
            u1=u1b,
        )
    )
    return "".join(bits)


def _fourier_planes(pngs: list[str], *, origin: np.ndarray, half: float) -> str:
    bits = [
        comment_block(
            "Fourier pooling: last-layer channel maps that enter FourierCodedPool2d "
            "(actual nodes, same colormap as the 64-channel volume)."
        )
    ]
    for i, png in enumerate(pngs):
        c = origin + np.array([i * 0.85, i * 0.50, 0.0])
        bits.append(
            box(
                c + np.array([-0.12, -half * 0.55, -half * 0.55]),
                c + np.array([0.12, half * 0.55, half * 0.55]),
                pigment="color rgb <0.14, 0.15, 0.17>",
            )
        )
        bits.append(
            _quad_neg_y(
                png,
                c + np.array([0.0, -half * 0.56, 0.0]),
                half * 0.52,
                half * 0.52,
            )
        )
    return "".join(bits)


def _flatten_slab(faces: dict[str, str], *, origin: np.ndarray) -> str:
    bits = [
        comment_block(
            "Flatten readout: large dense slab (64×128×128 activations) to "
            "emphasize parameter burden versus compact pooling."
        )
    ]
    hx, hy, hz = 3.2, 9.5, 6.5
    lo = origin + np.array([-hx, -hy, -hz])
    hi = origin + np.array([hx, hy, hz])
    bits.append(box(lo, hi, pigment="color rgb <0.12, 0.13, 0.15>"))
    bits.append(_cuboid_activation_skins(lo, hi, faces))
    return "".join(bits)


def _mlp_stack(*, origin: np.ndarray) -> tuple[str, float]:
    bits = [comment_block("MLP / localization head: layered dark blocks.")]
    widths = (5.5, 4.2, 3.0, 1.6)
    x = float(origin[0])
    for w in widths:
        bits.append(
            box(
                np.array([x, -w * 0.35, origin[2] - w * 0.28]),
                np.array([x + 1.35, w * 0.35, origin[2] + w * 0.28]),
                pigment="color rgb <0.10, 0.11, 0.13>",
            )
        )
        x += 2.9
    return "".join(bits), x


def _luminous_pred_ball(
    center: np.ndarray,
    *,
    radius: float,
    pigment: str,
    emission: str,
    light,
    comment: str,
    intensity: float = 0.5,
    fade_distance: float | None = None,
) -> str:
    """Emissive prediction marker with a weak co-located light_source."""
    c = np.asarray(center, dtype=float).reshape(3)
    r = float(radius)
    intensity = float(intensity)
    fd = max(2.8, r * 8.0) if fade_distance is None else float(fade_distance)
    bits = [comment_block(comment)]
    bits.append(
        sphere(
            c,
            r,
            pigment=f"color {pigment}",
            finish=(
                f"finish {{ ambient 0.12 diffuse 0.08 phong 0 emission {emission} }}"
            ),
            extra="no_shadow",
        )
    )
    if intensity <= 0.0:
        return "".join(bits)
    lr, lg, lb = (float(v) * intensity for v in np.asarray(light, dtype=float).reshape(3))
    bits.append(
        "light_source {\n"
        f"  {pov_vec(c)}\n"
        f"  color rgb <{lr:.4g}, {lg:.4g}, {lb:.4g}>\n"
        f"  fade_distance {fd:.6g}\n"
        "  fade_power 2\n"
        "}\n"
    )
    return "".join(bits)


def _localization_bear(
    *,
    origin: np.ndarray,
    bear_inc_name: str,
    mesh_centroid: np.ndarray,
    mesh_extent: float,
    catalog_radius: float,
    edge_solids: str = "",
    mesh_zmin: float | None = None,
    target_span: float = _LOCALIZATION_BEAR_SPAN,
    y_pred_fourier: np.ndarray | None = None,
    y_pred_pooled: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    comment: str | None = None,
) -> str:
    """Scaled transmissive bear; optional Fourier (blue), GAP (red), target (green)."""
    c = np.asarray(mesh_centroid, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    extent = max(float(mesh_extent), 1e-6)
    scale = float(target_span) / extent
    ox, oy = float(origin[0]), float(origin[1])
    cx, cy, cz = (float(v) for v in c)
    zmin = cz if mesh_zmin is None else float(mesh_zmin)
    oz = _FLOOR_Z + _BEAR_FLOOR_GAP - scale * (zmin - cz)
    origin = np.array([ox, oy, oz], dtype=float)
    ball_r = 0.55 * (float(target_span) / float(_LOCALIZATION_BEAR_SPAN))

    def _at(xyz: np.ndarray) -> np.ndarray:
        return origin + scale * (np.asarray(xyz, dtype=float).reshape(3) - c)

    bits = [
        comment_block(
            comment
            or (
                "Localization: scaled gummybear mesh (same pigment as the "
                "physical scene). Feet sit on the floor plane."
            )
        ),
        f'#include "{bear_inc_name}"\n',
        "union {\n",
        "  object {\n",
        "    BearMesh\n",
        "    pigment { color rgbt <0.68, 0.69, 0.71, 0.72> }\n",
        "    finish { ambient 0.06 diffuse 0.55 specular 0 roughness 0.5 "
        "reflection 0 }\n",
        "    interior { ior 1.02 }\n",
        "  }\n",
        str(edge_solids),
        f"  translate <-{cx:.6g}, {-cy:.6g}, {-cz:.6g}>\n",
        f"  scale {scale:.6g}\n",
        f"  translate <{ox:.6g}, {oy:.6g}, {oz:.6g}>\n",
        "}\n",
    ]
    if y_pred_fourier is not None:
        bits.append(
            _luminous_pred_ball(
                _at(y_pred_fourier),
                radius=ball_r,
                pigment="rgb <0.18, 0.42, 0.98>",
                emission="rgb <0.45, 0.72, 2.4>",
                light=np.array([0.28, 0.55, 2.2]),
                intensity=0.95,
                comment=(
                    "Fourier-path xyz prediction "
                    f"(catalog radius={catalog_radius:g})."
                ),
            )
        )
    if y_pred_pooled is not None:
        bits.append(
            _luminous_pred_ball(
                _at(y_pred_pooled),
                radius=ball_r,
                pigment="rgb <0.95, 0.16, 0.14>",
                emission="rgb <2.35, 0.38, 0.28>",
                light=np.array([2.2, 0.28, 0.22]),
                intensity=0.95,
                comment=(
                    "GAP/pooling-path xyz prediction "
                    f"(catalog radius={catalog_radius:g})."
                ),
            )
        )
    if y_true is not None:
        bits.append(
            _luminous_pred_ball(
                _at(y_true),
                radius=ball_r,
                pigment="rgb <0.14, 0.82, 0.28>",
                emission="rgb <0.45, 1.7, 0.55>",
                light=np.array([0.22, 1.85, 0.38]),
                comment=(
                    f"Ground-truth particle centre (catalog radius={catalog_radius:g})."
                ),
            )
        )
    return "".join(bits)


def build_network_pov_scene(
    bundle: NetworkActivationBundle,
    textures: dict[str, str | list[str]],
    *,
    yaw_deg: float = 80.0,
    fov_deg: float = 60.0,
    distance_scale: float = 1.05,
    input_stack_center: np.ndarray | None = None,
    input_stack_scale: float = 1.0,
    fourier_pane_yaw_deg: float = 0.0,
    fourier_embed_offset: tuple[float, float, float] | None = None,
    fourier_embed_scale: float = 1.0,
    fourier_group_offset: tuple[float, float, float] | None = None,
    gap_embed_yaw_deg: float = 0.0,
    gap_embed_scale: float = 1.0,
    gap_embed_shift: float = 0.0,
    cnn_fourier_front: float = 0.0,
    bear_inc_name: str | None = None,
    bear_mesh_centroid: np.ndarray | None = None,
    bear_mesh_extent: float | None = None,
    particle_catalog_radius: float = 0.5,
    bear_edge_solids: str = "",
    bear_mesh_zmin: float | None = None,
) -> str:
    """Assemble the schematic inference POV string."""
    half = FACE_HALF
    zc = CNN_Z_CENTER
    input_pngs = list(textures["input"])
    cnn_faces = list(textures["cnn"])
    cnn_gap_faces = list(textures["cnn_gap"])
    gap_png = str(textures["gap"])
    fourier_pane = str(textures["fourier_pane"])
    gap_pane = str(textures["gap_pane"])
    fourier_pool_png = str(textures["fourier_pool"])
    loc, look = illustration_camera_pose(yaw_deg=yaw_deg, distance_scale=distance_scale)
    stack_c = (
        DEFAULT_INPUT_STACK_CENTER
        if input_stack_center is None
        else np.asarray(input_stack_center, dtype=float).reshape(3)
    )
    input_half = float(half) * float(input_stack_scale)

    chunks = [_header(), _camera(yaw_deg=yaw_deg, fov_deg=fov_deg, distance_scale=distance_scale)]
    chunks.append(comment_block(f"Activation source: {bundle.source}."))
    chunks.append(_floor())
    chunks.append(
        _input_stack(
            input_pngs,
            center=stack_c,
            view_loc=loc,
            view_look=look,
            half=input_half,
        )
    )
    front_w = float(cnn_fourier_front) * _horiz_toward_camera(loc, look)
    toward = _horiz_toward_camera(loc, look)
    back_w = -front_w
    if float(np.linalg.norm(back_w)) < 1e-6:
        back_w = -12.0 * toward
    cnn_front = pipeline_to_world(np.array([13.0, 0.0, zc])) + front_w
    x_after = _cnn_end_x(start_x=13.0, depth_half=DEPTH_HALF)
    last_face_x = x_after - 3.8
    cnn_exit = pipeline_to_world(np.array([last_face_x, 0.0, zc]))
    gap_scale = float(gap_embed_scale)
    gap_l = _GAP_SLAB_LENGTH * gap_scale
    gap_r = _GAP_SLAB_RADIUS * gap_scale
    cnn_back = pipeline_to_world(np.array([13.0, 0.0, zc])) + back_w
    pool_lat = 1.5 * toward
    arrow_tail = np.asarray(stack_c, dtype=float) + toward * 6.0
    arrow_tip = cnn_front + toward * 6.0
    chunks.append(_inset_arrow(arrow_tail, arrow_tip))
    chunks.append(
        _inset_arrow(
            np.asarray(stack_c, dtype=float) - toward * 2.0,
            cnn_back + pool_lat,
            tail_frac=0.16,
            tip_frac=0.26,
            pigment=POOLING_ARROW_PIGMENT,
        )
    )
    group = (
        DEFAULT_FOURIER_GROUP_OFFSET
        if fourier_group_offset is None
        else (
            float(fourier_group_offset[0]),
            float(fourier_group_offset[1]),
            float(fourier_group_offset[2]),
        )
    )
    fourier_c = (
        pipeline_to_world(np.array([x_after + 10.0, 0.0, zc]))
        + np.array(group, dtype=float)
        + front_w
    )
    chunks.append(
        _input_stack(
            [fourier_pane],
            center=fourier_c,
            view_loc=loc,
            view_look=look,
            half=half,
            extra_yaw_deg=float(fourier_pane_yaw_deg),
            comment=(
                "Fourier term×activation plate on the CNN line: channels "
                "0, 9, 14, 61 (2×2). "
                f"Centre {pov_vec(fourier_c)}."
            ),
        )
    )
    gap_pane_c = (
        pipeline_to_world(np.array([x_after + 10.0, 0.0, zc]))
        + np.array(group, dtype=float)
        + back_w
    )
    chunks.append(
        _input_stack(
            [gap_pane],
            center=gap_pane_c,
            view_loc=loc,
            view_look=look,
            half=half,
            extra_yaw_deg=float(fourier_pane_yaw_deg),
            comment=(
                "GAP-branch last-layer maps (no Fourier multiplier): channels "
                "0, 9, 14, 61 (2×2), same layout as the Fourier pane. "
                f"Centre {pov_vec(gap_pane_c)}."
            ),
        )
    )
    chunks.append(
        _inset_arrow(
            cnn_exit + front_w + toward * 5.0,
            fourier_c + toward * 5.0,
            tail_frac=0.20,
            tip_frac=0.16,
            keep_head=True,
        )
    )
    chunks.append(
        _inset_arrow(
            cnn_exit + back_w + pool_lat,
            gap_pane_c + pool_lat,
            tail_frac=0.20,
            tip_frac=0.16,
            keep_head=True,
            pigment=POOLING_ARROW_PIGMENT,
        )
    )
    plate_fwd, plate_right, plate_up = _plate_axes(
        loc, look, extra_yaw_deg=float(fourier_pane_yaw_deg)
    )
    front_extra, right, up_from_bottom = (
        DEFAULT_FOURIER_EMBED_OFFSET
        if fourier_embed_offset is None
        else (
            float(fourier_embed_offset[0]),
            float(fourier_embed_offset[1]),
            float(fourier_embed_offset[2]),
        )
    )
    embed_r = _EMBED_SLAB_RADIUS * float(fourier_embed_scale)
    embed_l = 2.0 * float(half) * float(fourier_embed_scale)
    # front: toward camera; right: screen-right (−plate_right); up: along plate_up
    # from the spatial bottom of the pane.
    embed_origin = (
        fourier_c
        - (_PANE_HALF_THICK + embed_r + front_extra) * plate_fwd
        - right * plate_right
        - (float(half) - embed_r) * plate_up
        + up_from_bottom * plate_up
        - 0.5 * embed_l * plate_right
    )
    chunks.append(
        _oriented_pooling_slab(
            fourier_pool_png,
            origin=embed_origin,
            long_dir=plate_right,
            toward_camera=-plate_fwd,
            up_dir=plate_up,
            length=embed_l,
            radius=embed_r,
            comment=(
                "Fourier pooled embedding [64] as turbo bands along a long slab "
                "(same min/max as the inspection strip), just in front of the "
                "Fourier pane, offset right, at the spatial bottom."
            ),
        )
    )
    chunks.append(
        comment_block(
            "Pipeline solids rotated −90° about world z through the look-at "
            "so the flow reads left-to-right (not along the camera axis)."
        )
    )
    cnn_kw = dict(start_x=13.0, half=half, depth_half=DEPTH_HALF, zc=zc)
    cnn, x_after = _cnn_volumes(cnn_faces, **cnn_kw)
    chunks.append(_pipeline_union(cnn, extra_world=front_w))
    cnn_gap, x_gap_end = _cnn_volumes(
        cnn_gap_faces,
        **cnn_kw,
        comment=(
            "GAP-branch CNN volumes: same Encode 16→32→64 geometry as the "
            "Fourier branch; skins are the pooled-arch [C,H,W] maps."
        ),
    )
    gap_ox = float(x_gap_end) + 3.0 + float(gap_embed_shift)
    gap_cx = gap_ox + 0.5 * gap_l
    gap_body = (
        "union {\n"
        + _gap_slab(
            gap_png,
            origin=np.array([gap_ox, 0.0, zc]),
            length=gap_l,
            radius=gap_r,
            comment=(
                "Average pooling: GAP [64] as turbo bands on all six faces, "
                "on the pooling-branch CNN line."
            ),
        )
        + f"  translate <-{gap_cx:.6g}, 0, {-zc:.6g}>\n"
        + f"  rotate <0, 0, {float(gap_embed_yaw_deg):.6g}>\n"
        + f"  translate <{gap_cx:.6g}, 0, {zc:.6g}>\n"
        + "}\n"
    )
    chunks.append(_pipeline_union(cnn_gap + gap_body, extra_world=back_w))
    loc_pred = np.array([x_after + 58.0, 0.0, 4.0])
    loc_true = np.array([x_after + 66.0, 0.0, 4.0])
    if bear_inc_name:
        centroid = (
            np.zeros(3)
            if bear_mesh_centroid is None
            else np.asarray(bear_mesh_centroid, dtype=float).reshape(3)
        )
        extent = 20.0 if bear_mesh_extent is None else float(bear_mesh_extent)
        loc_kw = dict(
            bear_inc_name=str(bear_inc_name),
            mesh_centroid=centroid,
            mesh_extent=extent,
            catalog_radius=float(particle_catalog_radius),
            edge_solids=bear_edge_solids,
            mesh_zmin=bear_mesh_zmin,
        )
        pred_extra = -14.0 * toward
        chunks.append(
            _pipeline_union(
                _localization_bear(
                    origin=loc_pred,
                    target_span=_LOCALIZATION_BEAR_SPAN,
                    y_pred_fourier=bundle.y_pred,
                    y_pred_pooled=bundle.y_pred_pooled,
                    comment=(
                        "Prediction bear: Fourier (blue) and GAP/pooling (red) xyz. "
                        "Shifted right and back from the camera; feet on the floor."
                    ),
                    **loc_kw,
                ),
                extra_world=pred_extra,
            )
        )
        fourier_slab_w = embed_origin + 0.5 * embed_l * plate_right
        gap_end_p = np.array([gap_ox + gap_l, 0.0, zc], dtype=float)
        gap_pivot = np.array([gap_cx, 0.0, zc], dtype=float)
        gap_end_p = gap_pivot + _rot_z(gap_end_p - gap_pivot, float(gap_embed_yaw_deg))
        gap_slab_w = pipeline_to_world(gap_end_p) + back_w
        marker_kw = dict(
            origin_xy=loc_pred,
            mesh_centroid=centroid,
            mesh_extent=extent,
            mesh_zmin=bear_mesh_zmin,
            target_span=_LOCALIZATION_BEAR_SPAN,
        )
        fourier_ball_w = (
            pipeline_to_world(_bear_marker_pipeline(xyz=bundle.y_pred, **marker_kw))
            + pred_extra
        )
        gap_ball_w = (
            pipeline_to_world(
                _bear_marker_pipeline(xyz=bundle.y_pred_pooled, **marker_kw)
            )
            + pred_extra
        )
        chunks.append(
            comment_block(
                "Readout arrows: Fourier embedding slab to the blue prediction "
                "ball; GAP pooling slab to the red prediction ball."
            )
        )
        chunks.append(
            _inset_arrow(
                fourier_slab_w,
                fourier_ball_w,
                tail_frac=0.08,
                tip_frac=0.16,
                keep_head=True,
                pigment=FOURIER_TO_BALL_PIGMENT,
            )
        )
        chunks.append(
            _inset_arrow(
                gap_slab_w,
                gap_ball_w,
                tail_frac=0.08,
                tip_frac=0.16,
                keep_head=True,
                pigment=POOLING_ARROW_PIGMENT,
            )
        )
        if bundle.y_true is not None:
            chunks.append(
                _pipeline_union(
                    _localization_bear(
                        origin=loc_true,
                        target_span=0.5 * _LOCALIZATION_BEAR_SPAN,
                        y_true=bundle.y_true,
                        comment=(
                            "Target bear (50% scale): green ground-truth particle, "
                            "a bit toward the camera. Compare body location: blue "
                            "should match the target; red is a miss."
                        ),
                        **loc_kw,
                    ),
                    extra_world=5.0 * toward,
                )
            )
    return "".join(chunks)
