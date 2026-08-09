"""Geometry utilities for loading, summarizing, and validating STL triangle mesh files.

The public geometry API is re-exported here so callers can write::

    from gummybear.geometry import load_stl, describe_mesh, inspect_stl

Implementation lives in small purpose-specific modules:

- ``io.py`` — file loading and hashing
- ``summary.py`` — mesh summary dictionaries
- ``validation.py`` — projection-readiness checks
- ``mesh.py`` — mesh-derived sample points
- ``vector.py`` — vector normalization helpers
"""

from .io import load_stl, sha256_file
from .summary import describe_mesh, inspect_stl
from .validation import require_projection_ready, validate_mesh_for_projection
from .mesh import face_centroids
from .vector import normalize_vectors, normalize_vector


__all__ = [
    "load_stl",
    "sha256_file",
    "describe_mesh",
    "validate_mesh_for_projection",
    "require_projection_ready",
    "inspect_stl",
    "face_centroids",
    "normalize_vectors",
    "normalize_vector",
]
