"""Source-side ray bundles for illumination transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class RayBundleProtocol(Protocol):
    """Structural interface for ray bundles consumed by intersection helpers.

    Any object exposing these attributes satisfies the protocol; no inheritance
    is required.

    Attributes
    ----------
    origins : np.ndarray, shape (N, 3)
        Ray start points.
    directions : np.ndarray, shape (N, 3)
        Ray directions (normalized by consumers when needed).
    sample_shape : tuple[int, ...] or None
        Optional grid shape whose product equals ``N``.
    """

    origins: np.ndarray
    directions: np.ndarray
    sample_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class SourceRayBundle:
    """Weighted emission rays from a light source toward mesh entry points.

    Directions are normalized on construction. Weights are per-ray scalar
    intensities (non-negative, finite) used when depositing source energy.

    Attributes
    ----------
    origins : np.ndarray, shape (N, 3)
        Emission start points in world/mesh coordinates.
    directions : np.ndarray, shape (N, 3)
        Unit directions after validation.
    weights : np.ndarray, shape (N,)
        Non-negative per-ray weights.
    sample_shape : tuple[int, ...] or None, optional
        Optional grid shape; when set, ``N`` must equal its product.
    metadata : dict, optional
        Caller-defined sidecar (for example light index or sampling seed).
    """

    origins: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    sample_shape: tuple[int, ...] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins, dtype=float)
        directions = np.asarray(self.directions, dtype=float)
        weights = np.asarray(self.weights, dtype=float)

        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError(f"origins must be [N, 3], got {origins.shape}")
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(f"directions must be [N, 3], got {directions.shape}")
        if origins.shape != directions.shape:
            raise ValueError(
                f"origins/directions shape mismatch: {origins.shape} vs {directions.shape}"
            )
        if weights.ndim != 1 or len(weights) != len(origins):
            raise ValueError(f"weights must be [N], got {weights.shape}")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and non-negative")

        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms == 0):
            raise ValueError("directions contains zero-length vectors.")

        if self.sample_shape is not None:
            expected_n = int(np.prod(self.sample_shape))
            if origins.shape[0] != expected_n:
                raise ValueError(
                    f"Ray count {origins.shape[0]} does not match "
                    f"sample_shape {self.sample_shape} = {expected_n}"
                )

        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "directions", directions / norms[:, None])
        object.__setattr__(self, "weights", weights)

    @property
    def n_rays(self) -> int:
        """Return the number of rays in the bundle."""
        return len(self.origins)
