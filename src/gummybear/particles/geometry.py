"""Analytic spherical particle geometry and ray-segment intersection.

Particles are continuous-space inclusions: they do not remesh the phantom or
change the diffusion operator. Overlapping spheres are rejected at validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


DEFAULT_PATH_LENGTH_TOL = 1e-12
DEFAULT_OVERLAP_GAP_TOL = 0.0


class ParticleOverlapError(ValueError):
    """Raised when two analytic particles occupy overlapping volume.

    Geometric sphere overlap is physically impossible for distinct inclusions
    in this forward model. Callers must reject overlapping configurations.
    """


@dataclass(frozen=True)
class ParticleSphere:
    """Analytic spherical inclusion with absorb/scatter coefficients.

    Attributes:
        center: World-space centre ``(3,)`` in mm.
        radius: Sphere radius (mm, must be positive).
        mu_abs, mu_scat: Absorption coefficient and scattering coefficient inside the sphere.
        particle_id: Optional stable label for manifests and cache keys.

    See also:
        :class:`ParticleSet` — ordered group with overlap validation.
        :func:`~gummybear.particles.perturbation.build_affected_transport_pairs` — clean/dirty transport pairs.
    """

    center: np.ndarray
    radius: float
    mu_abs: float = 0.0
    mu_scat: float = 0.0
    particle_id: str | None = None

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float).reshape(3)
        radius = float(self.radius)
        mu_abs = float(self.mu_abs)
        mu_scat = float(self.mu_scat)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if mu_abs < 0.0 or mu_scat < 0.0:
            raise ValueError("mu_abs and mu_scat must be non-negative")
        if not np.all(np.isfinite(center)):
            raise ValueError("center must be finite")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "mu_abs", mu_abs)
        object.__setattr__(self, "mu_scat", mu_scat)

    @property
    def mu_total(self) -> float:
        """Total extinction coefficient (absorption plus scatter)."""
        return self.mu_abs + self.mu_scat

    def to_manifest_dict(self, index: int = 0) -> dict:
        """Return a JSON file-friendly particle record for sequence manifests."""
        pid = self.particle_id if self.particle_id is not None else f"p{index:03d}"
        return {
            "particle_id": pid,
            "shape": "sphere",
            "center": [float(x) for x in self.center],
            "radius": self.radius,
            "mu_abs": self.mu_abs,
            "mu_scat": self.mu_scat,
        }


@dataclass(frozen=True)
class ParticleSet:
    """Ordered collection of analytic particles (workbook order is scientific).

    Attributes:
        particles: Non-empty or empty tuple of :class:`ParticleSphere` instances.
        metadata: Optional free-form manifest metadata (placement mode, group id).

    See also:
        :meth:`validate` / :meth:`require_valid` — reject geometric overlap.
        :func:`~gummybear.particles.perturbation.build_affected_transport_pairs` — transport pairs per particle.
    """

    particles: tuple[ParticleSphere, ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        particles = tuple(self.particles)
        object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __len__(self) -> int:
        """Return the number of particles in workbook order."""
        return len(self.particles)

    def __iter__(self):
        return iter(self.particles)

    def __getitem__(self, index: int) -> ParticleSphere:
        """Return the particle at ``index`` (0-based workbook order)."""
        return self.particles[index]

    def overlapping_pairs(
        self,
        *,
        gap_tol: float = DEFAULT_OVERLAP_GAP_TOL,
    ) -> list[tuple[int, int, float, float]]:
        """Return overlapping index pairs ``(i, j, distance, min_center_sep)``.

        Spheres *i* and *j* overlap when
        ``distance(centers) < radius_i + radius_j - gap_tol``.
        Touching at a point (``distance == r_i + r_j``) is allowed when
        ``gap_tol == 0``.
        """
        if gap_tol < 0.0:
            raise ValueError("gap_tol must be non-negative")
        overlaps: list[tuple[int, int, float, float]] = []
        n = len(self.particles)
        for i in range(n):
            a = self.particles[i]
            for j in range(i + 1, n):
                b = self.particles[j]
                distance = float(np.linalg.norm(a.center - b.center))
                min_sep = float(a.radius + b.radius) - float(gap_tol)
                if distance < min_sep:
                    overlaps.append((i, j, distance, min_sep))
        return overlaps

    def validate(self, *, gap_tol: float = DEFAULT_OVERLAP_GAP_TOL) -> bool:
        """Return True when no two particles occupy overlapping volume."""
        return not self.overlapping_pairs(gap_tol=gap_tol)

    def require_valid(
        self,
        *,
        gap_tol: float = DEFAULT_OVERLAP_GAP_TOL,
    ) -> ParticleSet:
        """Return ``self`` or raise ``ParticleOverlapError`` on overlap."""
        overlaps = self.overlapping_pairs(gap_tol=gap_tol)
        if not overlaps:
            return self
        details = []
        for i, j, distance, min_sep in overlaps:
            a = self.particles[i]
            b = self.particles[j]
            a_id = a.particle_id if a.particle_id is not None else f"index:{i}"
            b_id = b.particle_id if b.particle_id is not None else f"index:{j}"
            details.append(
                f"{a_id!r} vs {b_id!r}: distance={distance:.6g} "
                f"< required_separation={min_sep:.6g}"
            )
        raise ParticleOverlapError(
            "ParticleSet contains geometrically overlapping spheres "
            "(physically impossible for distinct inclusions): "
            + "; ".join(details)
        )

    @classmethod
    def from_particles(
        cls,
        particles: Sequence[ParticleSphere],
        *,
        metadata: dict | None = None,
        require_non_overlapping: bool = True,
        gap_tol: float = DEFAULT_OVERLAP_GAP_TOL,
    ) -> ParticleSet:
        """Build a ParticleSet; reject overlapping spheres by default."""
        particle_set = cls(particles=tuple(particles), metadata=dict(metadata or {}))
        if require_non_overlapping:
            particle_set.require_valid(gap_tol=gap_tol)
        return particle_set

    def to_manifest_block(self) -> dict:
        """Return the standard multi-particle block for sequence manifests."""
        return {
            "perturbation_mode": "ray_weight_and_source_deposition",
            "requires_remeshing": False,
            "changes_diffusion_operator": False,
            "particle_model": "analytic_spheres_absorb_scatter_v1",
            "count": len(self.particles),
            "items": [p.to_manifest_dict(i) for i, p in enumerate(self.particles)],
        }


@dataclass(frozen=True)
class ParticleIntersectionEvent:
    """One clipped segment–sphere overlap chord.

    Attributes:
        segment_index: Index into the segment ``starts``/``ends`` arrays.
        particle_index: Index into the :class:`ParticleSet`.
        entry_t, exit_t: Segment parameter range ``[0, 1]`` inside the sphere.
        entry_point, exit_point: World-space chord endpoints.
        path_length_inside_particle: Euclidean length of the chord (mm).
        midpoint_inside_particle: Chord midpoint (diagnostic / fallback assignment).
    """

    segment_index: int
    particle_index: int
    entry_t: float
    exit_t: float
    entry_point: np.ndarray
    exit_point: np.ndarray
    path_length_inside_particle: float
    midpoint_inside_particle: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "entry_point", np.asarray(self.entry_point, dtype=float).reshape(3)
        )
        object.__setattr__(
            self, "exit_point", np.asarray(self.exit_point, dtype=float).reshape(3)
        )
        object.__setattr__(
            self,
            "midpoint_inside_particle",
            np.asarray(self.midpoint_inside_particle, dtype=float).reshape(3),
        )


def segment_sphere_intersection(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    particle_center: np.ndarray,
    particle_radius: float,
    *,
    segment_index: int = 0,
    particle_index: int = 0,
    path_length_tol: float = DEFAULT_PATH_LENGTH_TOL,
) -> ParticleIntersectionEvent | None:
    """Intersect a finite segment with a sphere.

    Parameterisation::

        P(t) = segment_start + t * (segment_end - segment_start),  t in [0, 1]

    Args:
        segment_start, segment_end: Segment endpoints ``(3,)``.
        particle_center: Sphere centre ``(3,)``.
        particle_radius: Sphere radius (mm).
        segment_index, particle_index: Indices recorded on the event.
        path_length_tol: Ignore chords shorter than this length.

    Returns:
        ParticleIntersectionEvent for a non-degenerate chord, else ``None``.
    """
    p0 = np.asarray(segment_start, dtype=float).reshape(3)
    p1 = np.asarray(segment_end, dtype=float).reshape(3)
    c = np.asarray(particle_center, dtype=float).reshape(3)
    r = float(particle_radius)
    if r <= 0.0:
        raise ValueError("particle_radius must be positive")

    d = p1 - p0
    seg_len = float(np.linalg.norm(d))
    if seg_len <= path_length_tol:
        return None

    # Quadratic in t for |P(t) - c|^2 = r^2
    f = p0 - c
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(f, d))
    cc = float(np.dot(f, f)) - r * r
    disc = b * b - 4.0 * a * cc

    if disc < 0.0:
        # Entirely outside, or start inside with numerical miss — check containment.
        if cc <= 0.0:
            # Start inside; end also inside (no real exit root) → full segment.
            entry_t, exit_t = 0.0, 1.0
        else:
            return None
    else:
        sqrt_disc = float(np.sqrt(max(disc, 0.0)))
        inv_2a = 0.5 / a
        t0 = (-b - sqrt_disc) * inv_2a
        t1 = (-b + sqrt_disc) * inv_2a
        if t0 > t1:
            t0, t1 = t1, t0

        # Clip to segment domain.
        entry_t = max(t0, 0.0)
        exit_t = min(t1, 1.0)

        if exit_t <= entry_t:
            # No overlap with [0, 1], unless start is inside (cc <= 0) and
            # the outward root is beyond the segment start.
            if cc <= 0.0 and t1 > 0.0:
                entry_t = 0.0
                exit_t = min(t1, 1.0)
            else:
                return None

        if exit_t <= entry_t:
            return None

    entry_point = p0 + entry_t * d
    exit_point = p0 + exit_t * d
    path_length = float(np.linalg.norm(exit_point - entry_point))
    if path_length <= path_length_tol:
        return None

    midpoint = 0.5 * (entry_point + exit_point)
    return ParticleIntersectionEvent(
        segment_index=int(segment_index),
        particle_index=int(particle_index),
        entry_t=float(entry_t),
        exit_t=float(exit_t),
        entry_point=entry_point,
        exit_point=exit_point,
        path_length_inside_particle=path_length,
        midpoint_inside_particle=midpoint,
    )


def intersect_segments_with_particles(
    starts: np.ndarray,
    ends: np.ndarray,
    particles: ParticleSet | Sequence[ParticleSphere],
    *,
    path_length_tol: float = DEFAULT_PATH_LENGTH_TOL,
) -> list[ParticleIntersectionEvent]:
    """Intersect many segments with many particles.

    Sort key: ``(segment_index, entry_t, particle_index)``.

    Args:
        starts, ends: Segment endpoint arrays with shape ``[N, 3]``.
        particles: :class:`ParticleSet` or sequence of spheres.
        path_length_tol: Minimum chord length to retain.

    Returns:
        Sorted list of intersection events (possibly empty).
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("starts/ends must share shape [N, 3]")

    if isinstance(particles, ParticleSet):
        particle_list = list(particles.particles)
    else:
        particle_list = list(particles)

    events: list[ParticleIntersectionEvent] = []
    for seg_i in range(len(starts)):
        for p_i, particle in enumerate(particle_list):
            ev = segment_sphere_intersection(
                starts[seg_i],
                ends[seg_i],
                particle.center,
                particle.radius,
                segment_index=seg_i,
                particle_index=p_i,
                path_length_tol=path_length_tol,
            )
            if ev is not None:
                events.append(ev)

    events.sort(key=lambda e: (e.segment_index, e.entry_t, e.particle_index))
    return events
