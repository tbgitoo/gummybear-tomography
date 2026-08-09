"""Unit-vector normalization helpers for 3D geometry."""

import numpy as np


def normalize_vector(v, eps=1e-12):
    """Return a unit-length copy of one 3D vector.

    Raises ``ValueError`` when the input norm is below ``eps`` so callers do
    not silently propagate near-zero directions into ray or camera code.

    Parameters
    ----------
    v : array-like, shape (3,)
        Input vector in mesh/world coordinates.
    eps : float, optional
        Minimum acceptable norm before rejecting the vector.

    Returns
    -------
    np.ndarray, shape (3,)
        Normalized vector with the same dtype as the converted input.
    """
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")

    return v / norm


def normalize_vectors(v, eps=1e-12):
    """Return unit-length copies of many 3D vectors along the last axis.

    Unlike :func:`normalize_vector`, norms below ``eps`` are clamped to ``eps``
    rather than raising, which keeps batched ray-direction normalization stable
    when a few entries are numerically tiny.

    Parameters
    ----------
    v : array-like, shape (..., 3)
        Batch of vectors to normalize.
    eps : float, optional
        Floor applied to each norm before division.

    Returns
    -------
    np.ndarray, shape (..., 3)
        Normalized vectors with the same leading shape as ``v``.
    """
    v = np.asarray(v, dtype=float)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norms, eps)
