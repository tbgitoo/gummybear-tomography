"""STL triangle mesh file loading and content hashing for mesh interchange."""

from pathlib import Path
import hashlib

import numpy as np
import trimesh


def load_stl(path):
    """Load an STL triangle mesh file as a ``trimesh.Trimesh`` without repair or rescaling.

    The returned mesh preserves vertex coordinates and face topology exactly
    as stored in the file. Callers that need watertight or projection-ready
    geometry should validate separately.

    Parameters
    ----------
    path : str or pathlib.Path
        Filesystem path to an ``.stl`` STL triangle mesh file.

    Returns
    -------
    trimesh.Trimesh
        Loaded triangle mesh.

    Raises
    ------
    ValueError
        If the path is missing or trimesh cannot parse the STL triangle mesh file.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"STL file not found: {path}")

    try:
        mesh = trimesh.load(path, force="mesh")
    except Exception as exc:
        raise ValueError(f"Failed to load STL file: {path}") from exc

    return mesh


def sha256_file(path):
    """Return the lowercase hex SHA-256 digest of a file's raw bytes.

    Useful for pinning mesh provenance in summaries and manifests.

    Parameters
    ----------
    path : str or pathlib.Path
        File to hash.

    Returns
    -------
    str
        64-character hexadecimal digest.
    """
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def face_centroids(mesh):
    """Return the centroid of each triangle face.

    Centroids are the arithmetic mean of the three corner vertices and are
    commonly used as face sample points for illumination and transport.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh whose ``triangles`` attribute holds corner coordinates.

    Returns
    -------
    np.ndarray, shape (n_faces, 3)
        Face centroid coordinates in the mesh coordinate frame.
    """
    triangles = np.asarray(mesh.triangles)
    return triangles.mean(axis=1)
