"""POV-Ray 3.7 primitive snippets. Coordinates are simulation millimetres."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pov_vec(xyz: Sequence[float] | np.ndarray) -> str:
    x, y, z = (float(v) for v in np.asarray(xyz, dtype=float).reshape(3))
    return f"<{x:.6g}, {y:.6g}, {z:.6g}>"


def comment_block(text: str) -> str:
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    return "\n".join(f"// {ln}" if ln else "//" for ln in lines) + "\n"


def sphere(
    center,
    radius: float,
    *,
    pigment: str,
    finish: str = "finish { phong 0.3 }",
    extra: str = "",
) -> str:
    extra_block = f"  {extra.strip()}\n" if extra.strip() else ""
    return (
        f"sphere {{ {pov_vec(center)}, {float(radius):.6g}\n"
        f"  pigment {{ {pigment} }}\n"
        f"  {finish}\n"
        f"{extra_block}"
        f"}}\n"
    )


def cylinder(
    start,
    end,
    radius: float,
    *,
    pigment: str,
    extra: str = "no_shadow",
    finish: str = "finish { ambient 0.08 diffuse 0.55 }",
) -> str:
    extra_block = f"  {extra.strip()}\n" if extra.strip() else ""
    return (
        f"cylinder {{ {pov_vec(start)}, {pov_vec(end)}, {float(radius):.6g}\n"
        f"  pigment {{ {pigment} }}\n"
        f"  {finish}\n"
        f"{extra_block}"
        f"}}\n"
    )


def box(corner_a, corner_b, *, pigment: str) -> str:
    return (
        f"box {{ {pov_vec(corner_a)}, {pov_vec(corner_b)}\n"
        f"  pigment {{ {pigment} }}\n"
        f"  finish {{ phong 0.2 }}\n"
        f"}}\n"
    )


def cone(
    start,
    start_radius: float,
    end,
    end_radius: float,
    *,
    pigment: str,
    extra: str = "no_shadow hollow",
    finish: str = "finish { ambient 0.7 diffuse 0.1 }",
) -> str:
    extra_block = f"  {extra.strip()}\n" if extra.strip() else ""
    return (
        f"cone {{ {pov_vec(start)}, {float(start_radius):.6g}, "
        f"{pov_vec(end)}, {float(end_radius):.6g}\n"
        f"  pigment {{ {pigment} }}\n"
        f"  {finish}\n"
        f"{extra_block}"
        f"}}\n"
    )


def plane_z(z0: float, *, pigment: str, finish: str) -> str:
    return (
        f"plane {{ z, {float(z0):.6g}\n"
        f"  pigment {{ {pigment} }}\n"
        f"  {finish}\n"
        f"}}\n"
    )
