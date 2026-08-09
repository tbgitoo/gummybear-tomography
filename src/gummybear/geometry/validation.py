"""Mesh validation helpers for projection readiness."""


def validate_mesh_for_projection(mesh):
    """Report whether a mesh is suitable for ray-based path-length projection.

    Performs read-only checks only; it does not repair geometry. Watertightness
    is treated as a hard requirement because enter/exit ray pairing needs a
    closed surface.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to validate.

    Returns
    -------
    dict
        ``usable_for_projection`` (bool), ``errors`` (list of str), and
        ``warnings`` (list of str).
    """
    errors = []
    warnings = []

    if len(mesh.vertices) == 0:
        errors.append("Mesh has no vertices.")

    if len(mesh.faces) == 0:
        errors.append("Mesh has no faces.")

    if any(length <= 0 for length in mesh.extents):
        errors.append(
            "Mesh has a degenerate bounding box. At least one axis has zero extent."
        )

    if not mesh.is_watertight:
        errors.append(
            "Mesh is not watertight. Path-length projection requires closed surfaces "
            "for reliable enter/exit ray pairing."
        )

    if not mesh.is_winding_consistent:
        warnings.append(
            "Mesh winding is not consistent. This may affect later ray-facing logic."
        )

    return {
        "usable_for_projection": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def require_projection_ready(mesh):
    """Raise ``ValueError`` when the mesh fails projection-readiness checks.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to validate.

    Returns
    -------
    None
        Returns ``None`` when validation passes.

    Raises
    ------
    ValueError
        When :func:`validate_mesh_for_projection` reports blocking errors.
    """
    result = validate_mesh_for_projection(mesh)

    if not result["usable_for_projection"]:
        joined = "\n".join(f"- {msg}" for msg in result["errors"])
        raise ValueError(f"Mesh is not usable for projection:\n{joined}")

    return None
