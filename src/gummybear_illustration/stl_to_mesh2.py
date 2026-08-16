"""Convert an STL triangle mesh to a POV-Ray 3.7 ``mesh2`` include file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def write_stl_mesh2_inc(
    stl_path: str | Path,
    inc_path: str | Path,
    *,
    object_name: str = "BearMesh",
) -> Path:
    """Write ``#declare BearMesh = mesh2 { ... }`` from an STL.

    Vertices keep the STL coordinates (simulation millimetres). No repair or
    rescale. Large phantoms produce large ``.inc`` files; keep them under
    ``outputs/`` (not the git tree).
    """
    stl_path = Path(stl_path)
    inc_path = Path(inc_path)
    mesh = trimesh.load(stl_path, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Unexpected vertex array shape {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected face array shape {faces.shape}")
    inc_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "// Auto-generated mesh2 from STL. Coordinates = simulation millimetres.",
        f"// Source: {stl_path.as_posix()}",
        f"#declare {object_name} = mesh2 {{",
        f"  vertex_vectors {{ {len(vertices)},",
    ]
    for v in vertices:
        lines.append(f"    <{v[0]:.8g}, {v[1]:.8g}, {v[2]:.8g}>,")
    lines.append("  }")
    lines.append(f"  face_indices {{ {len(faces)},")
    for f in faces:
        lines.append(f"    <{int(f[0])}, {int(f[1])}, {int(f[2])}>,")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    inc_path.write_text("\n".join(lines), encoding="utf-8")
    return inc_path
