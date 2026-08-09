"""Mesh-derived geometric sample points."""

import numpy as np


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
