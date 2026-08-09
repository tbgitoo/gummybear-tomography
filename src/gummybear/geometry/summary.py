"""Deterministic mesh inspection summaries for logging and notebooks."""

from gummybear.paths import repo_relative_path

from .io import load_stl, sha256_file
from .validation import validate_mesh_for_projection


def describe_mesh(mesh, source_path=None, units="mm"):
    """Build a JSON file-serializable summary dictionary for one mesh.

    Captures counts, axis-aligned bounds, centroid, surface area, volume, and
    basic topology flags. When ``source_path`` is supplied, the summary also
    records a repository-relative path and content hash for reproducibility.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to summarize.
    source_path : str, pathlib.Path, or None, optional
        Original STL triangle mesh file path used to load ``mesh``. When provided, adds
        ``source_path`` and ``stl_sha256`` keys.
    units : str, optional
        Coordinate-unit label stored in the summary (default ``"mm"``).

    Returns
    -------
    dict
        Keys include ``vertices``, ``faces``, ``bbox_min``, ``bbox_max``,
        ``bbox_size``, ``centroid``, ``area``, ``volume``, ``is_watertight``,
        ``is_winding_consistent``, and ``units``.
    """
    summary = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bbox_min": [float(x) for x in mesh.bounds[0]],
        "bbox_max": [float(x) for x in mesh.bounds[1]],
        "bbox_size": [float(x) for x in mesh.extents],
        "centroid": [float(x) for x in mesh.centroid],
        "area": float(mesh.area),
        "volume": float(mesh.volume),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "units": units,
    }

    if source_path is not None:
        summary["source_path"] = repo_relative_path(source_path)
        summary["stl_sha256"] = sha256_file(source_path)

    return summary


def inspect_stl(path, units="mm"):
    """Load an STL triangle mesh file and return mesh, summary, and projection-readiness report.

    Convenience wrapper that combines :func:`load_stl`, :func:`describe_mesh`,
    and :func:`validate_mesh_for_projection` for interactive inspection.

    Parameters
    ----------
    path : str or pathlib.Path
        STL triangle mesh file to load.
    units : str, optional
        Coordinate-unit label passed to :func:`describe_mesh`.

    Returns
    -------
    dict
        Mapping with keys ``mesh`` (``trimesh.Trimesh``), ``summary`` (dict),
        and ``validation`` (dict from
        :func:`validate_mesh_for_projection`).
    """
    mesh = load_stl(path)
    summary = describe_mesh(mesh, source_path=path, units=units)
    validation = validate_mesh_for_projection(mesh)
    return {
        "mesh": mesh,
        "summary": summary,
        "validation": validation,
    }
