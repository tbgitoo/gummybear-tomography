"""Clean/dirty transport pairs and particle source correction.

Particles perturb dirty transport relative to clean paths. The accounting
object is an affected clean/dirty pair; intersection events are construction
inputs. Default scatter deposition uses attenuated chords (Beer–Lambert along
the in-particle segment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from .geometry import (
    DEFAULT_PATH_LENGTH_TOL,
    ParticleIntersectionEvent,
    ParticleSet,
    ParticleSphere,
    intersect_segments_with_particles,
)
from ..optics.diffusion_mesh import DiffusionMesh
from ..optics.material import OpticalMaterialConfig
from ..optics.source_deposition import (
    RaySegmentBundle,
    SourceDepositionResult,
    deposit_ray_source,
    distribute_energy_along_segment,
)


PARTITION_MODEL = "relative_mu_share_v1"
"""Label for splitting particle extinction into absorbed vs scattered shares."""


@dataclass(frozen=True)
class TransportInterval:
    """One ordered interval on a clean or dirty transport path.

    Attributes:
        path_id: Transport path identifier (``RaySegmentBundle.ray_ids``).
        segment_index: Source segment row index in the bundle.
        interval_id: Order within the path's interval list.
        s0, s1: Cumulative distance along the path (mm).
        start, end: World-space interval endpoints.
        I_in, I_out: Ballistic intensities entering and leaving the interval.
        active_particle_indices: Particle indices when ``role`` is particle.
        metadata: Construction tags (``role``, ``transport`` clean/dirty).
    """

    path_id: int
    segment_index: int
    interval_id: int
    s0: float
    s1: float
    start: np.ndarray
    end: np.ndarray
    I_in: float
    I_out: float
    active_particle_indices: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_id", int(self.path_id))
        object.__setattr__(self, "segment_index", int(self.segment_index))
        object.__setattr__(self, "interval_id", int(self.interval_id))
        object.__setattr__(self, "start", np.asarray(self.start, dtype=float).reshape(3))
        object.__setattr__(self, "end", np.asarray(self.end, dtype=float).reshape(3))
        object.__setattr__(
            self,
            "active_particle_indices",
            tuple(int(i) for i in self.active_particle_indices),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.s1 < self.s0:
            raise ValueError("TransportInterval requires s1 >= s0")
        if self.I_in < 0.0 or self.I_out < 0.0:
            raise ValueError("TransportInterval intensities must be non-negative")

    @property
    def length(self) -> float:
        """Euclidean distance between interval ``start`` and ``end`` (mm)."""
        return float(np.linalg.norm(self.end - self.start))


@dataclass(frozen=True)
class ParticleScatterSourceEvent:
    """Particle energy bookkeeping from one dirty in-particle interval.

    ``entry_point`` and ``exit_point`` bound chord-length scatter deposition;
    ``point`` is the chord midpoint for diagnostics and midpoint fallbacks.

    Attributes:
        path_id, segment_index, interval_id: Transport lineage coordinates.
        particle_index: Index into the :class:`ParticleSet`.
        entry_point, exit_point, point: Chord geometry in world space.
        E_scat, E_abs: Scattered and absorbed energy partitioned from extinction.
        I_before, I_after: Dirty intensities at chord entry and exit.
        metadata: Optical coefficients and path length, etc.
    """

    path_id: int
    segment_index: int
    interval_id: int
    particle_index: int
    entry_point: np.ndarray
    exit_point: np.ndarray
    point: np.ndarray
    E_scat: float
    E_abs: float
    I_before: float
    I_after: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_id", int(self.path_id))
        object.__setattr__(self, "segment_index", int(self.segment_index))
        object.__setattr__(self, "interval_id", int(self.interval_id))
        object.__setattr__(self, "particle_index", int(self.particle_index))
        object.__setattr__(
            self, "entry_point", np.asarray(self.entry_point, dtype=float).reshape(3)
        )
        object.__setattr__(
            self, "exit_point", np.asarray(self.exit_point, dtype=float).reshape(3)
        )
        object.__setattr__(self, "point", np.asarray(self.point, dtype=float).reshape(3))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.E_scat < 0.0 or self.E_abs < 0.0:
            raise ValueError("Particle source-event energies must be non-negative")

    @property
    def E_loss(self) -> float:
        """Total extinction energy ``E_abs + E_scat`` for this chord event."""
        return self.E_abs + self.E_scat

    @property
    def chord_length(self) -> float:
        """In-particle path length from ``entry_point`` to ``exit_point`` (mm)."""
        return float(np.linalg.norm(self.exit_point - self.entry_point))


@dataclass(frozen=True)
class AffectedTransportPair:
    """Clean and particle-dirty transport for one affected path.

    Attributes:
        path_id: Transport path identifier.
        clean_intervals: Background intervals on the unperturbed path.
        dirty_intervals: Replacement-model intervals including particle chords.
        particle_events: Geometric intersection events on this path.
        particle_scatter_source_events: Energy events derived from dirty intervals.
        clean_output_intensity, dirty_output_intensity: Exit intensities along path.
        total_E_abs, total_E_scat: Summed particle absorption and scatter.
        metadata: Partition model and background mu tags.
    """

    path_id: int
    clean_intervals: tuple[TransportInterval, ...]
    dirty_intervals: tuple[TransportInterval, ...]
    particle_events: tuple[ParticleIntersectionEvent, ...]
    particle_scatter_source_events: tuple[ParticleScatterSourceEvent, ...]
    clean_output_intensity: float
    dirty_output_intensity: float
    total_E_abs: float
    total_E_scat: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_id", int(self.path_id))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def total_E_loss(self) -> float:
        """Summed particle absorption and scatter on this path."""
        return self.total_E_abs + self.total_E_scat

    @property
    def delta_output_intensity(self) -> float:
        """Dirty minus clean exit intensity along the affected path."""
        return self.dirty_output_intensity - self.clean_output_intensity


@dataclass(frozen=True)
class AffectedTransportPairResult:
    """Collection of clean/dirty transport pairs for all affected paths.

    Attributes:
        pairs: One :class:`AffectedTransportPair` per affected ``path_id``.
        affected_path_ids: Sorted affected path identifiers.
        affected_segment_indices: Segment rows with at least one intersection.
        total_E_abs, total_E_scat: Aggregates across all pairs.
        metadata: Construction bookkeeping (overlap policy, path identity fields).
    """

    pairs: tuple[AffectedTransportPair, ...]
    affected_path_ids: tuple[int, ...]
    affected_segment_indices: np.ndarray
    total_E_abs: float
    total_E_scat: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "affected_segment_indices",
            np.asarray(self.affected_segment_indices, dtype=int),
        )
        object.__setattr__(
            self, "affected_path_ids", tuple(int(i) for i in self.affected_path_ids)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def total_E_loss(self) -> float:
        """Aggregate ``E_abs + E_scat`` across all affected pairs."""
        return self.total_E_abs + self.total_E_scat


@dataclass(frozen=True)
class ParticleScatterDepositionResult:
    """Element-integrated deposition of dirty-transport particle scattering.

    Attributes:
        delta_E_particle_scat_elem: Per-tet scatter energy delta array.
        assignment_mode: Deposition policy (``attenuated_chord``, etc.).
        outside_policy: Behaviour when chords miss the diffusion mesh.
        n_events, n_assigned: Event counts for accounting checks.
        total_E_scat_events, total_E_scat_deposited: Integrated scatter totals.
        metadata: Assignment counts and distribution label.
    """

    delta_E_particle_scat_elem: np.ndarray
    assignment_mode: str
    outside_policy: str
    n_events: int
    n_assigned: int
    total_E_scat_events: float
    total_E_scat_deposited: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta_E_particle_scat_elem",
            np.asarray(self.delta_E_particle_scat_elem, dtype=float),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TransportSourceCorrectionResult:
    """Diffusion source correction from dirty-minus-clean path contributions.

    ``E_particle_elem = E_clean_elem + delta_E_transport_elem`` with
    ``delta_E_transport_elem = delta_E_background_elem + delta_E_particle_scat_elem``.

    Attributes:
        E_clean_elem: Clean background scatter deposition (reference).
        delta_E_background_elem: Dirty-minus-clean background scatter delta.
        delta_E_particle_scat_elem: Particle scatter source deposition.
        delta_E_transport_elem: Sum of background and particle scatter deltas.
        E_particle_elem: Corrected element energy for particle diffusion solve.
        S_particle: Element source density derived from ``E_particle_elem``.
        clean_affected_deposition, dirty_affected_deposition: Per-pair background deposits.
        particle_scatter_deposition: Scatter-only deposition detail.
        metadata: Model tags and pair counts.
    """

    E_clean_elem: np.ndarray
    delta_E_background_elem: np.ndarray
    delta_E_particle_scat_elem: np.ndarray
    delta_E_transport_elem: np.ndarray
    E_particle_elem: np.ndarray
    S_particle: np.ndarray
    clean_affected_deposition: SourceDepositionResult
    dirty_affected_deposition: SourceDepositionResult
    particle_scatter_deposition: ParticleScatterDepositionResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "E_clean_elem",
            "delta_E_background_elem",
            "delta_E_particle_scat_elem",
            "delta_E_transport_elem",
            "E_particle_elem",
            "S_particle",
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        object.__setattr__(self, "metadata", dict(self.metadata))


def partition_particle_loss(
    E_loss_total: float,
    mu_abs: float,
    mu_scat: float,
) -> tuple[float, float]:
    """Partition extinction loss into absorbed and scattered relative-mu shares.

    Args:
        E_loss_total: Non-negative total extinction energy.
        mu_abs, mu_scat: Particle absorption and scatter coefficients.

    Returns:
        Tuple ``(E_abs, E_scat)``.
    """
    loss = float(E_loss_total)
    mu_a = float(mu_abs)
    mu_s = float(mu_scat)
    if loss < 0.0:
        raise ValueError("E_loss_total must be non-negative")
    if mu_a < 0.0 or mu_s < 0.0:
        raise ValueError("mu_abs and mu_scat must be non-negative")
    mu_total = mu_a + mu_s
    if mu_total <= 0.0 or loss == 0.0:
        return 0.0, 0.0
    return loss * (mu_a / mu_total), loss * (mu_s / mu_total)


def _as_particle_set(
    particles: ParticleSet | Sequence[ParticleSphere],
) -> ParticleSet:
    """Normalize to ParticleSet.

    Sequences are validated for non-overlap by default. An existing
    ``ParticleSet`` is returned as-is (callers that bypassed validation,
    e.g. interval-overlap fixtures, must opt in via
    ``from_particles(..., require_non_overlapping=False)``).
    """
    if isinstance(particles, ParticleSet):
        return particles
    return ParticleSet.from_particles(particles)


def _validate_events(
    segments: RaySegmentBundle,
    particles: ParticleSet,
    events: Sequence[ParticleIntersectionEvent],
) -> None:
    for event in events:
        if event.segment_index < 0 or event.segment_index >= segments.n_segments:
            raise IndexError(
                f"Particle event segment_index {event.segment_index} is out of range"
            )
        if event.particle_index < 0 or event.particle_index >= len(particles):
            raise IndexError(
                f"Particle event particle_index {event.particle_index} is out of range"
            )


def _resolve_background_mu(
    material: OpticalMaterialConfig | None,
    mu_s: float | None,
    mu_a: float | None,
) -> tuple[float, float, float]:
    """Return scattering coefficient ``mu_s``, absorption coefficient ``mu_a``, and ``mu_total`` for dirty-path background intervals."""
    if material is not None:
        mu_s_v = float(material.mu_scatter if mu_s is None else mu_s)
        mu_a_v = float(material.mu_absorption if mu_a is None else mu_a)
    else:
        if mu_s is None and mu_a is None:
            return 0.0, 0.0, 0.0
        if mu_s is None or mu_a is None:
            raise ValueError("Provide material or both mu_s and mu_a")
        mu_s_v = float(mu_s)
        mu_a_v = float(mu_a)
    if mu_s_v < 0.0 or mu_a_v < 0.0:
        raise ValueError("mu_s and mu_a must be non-negative")
    return mu_s_v, mu_a_v, mu_s_v + mu_a_v


def _background_interval_exit_intensity(
    I_in: float,
    length: float,
    mu_total: float,
) -> float:
    """Ballistic intensity after a dirty background interval of given length."""
    if length <= 0.0 or I_in == 0.0 or mu_total <= 0.0:
        return float(I_in)
    return float(I_in * np.exp(-mu_total * length))


def build_affected_transport_pairs(
    segments: RaySegmentBundle,
    particles: ParticleSet | Sequence[ParticleSphere],
    *,
    events: Sequence[ParticleIntersectionEvent] | None = None,
    path_length_tol: float = DEFAULT_PATH_LENGTH_TOL,
    material: OpticalMaterialConfig | None = None,
    mu_s: float | None = None,
    mu_a: float | None = None,
) -> AffectedTransportPairResult:
    """Construct clean/dirty transport pairs from ordered path lineage.

    ``segment_index`` always identifies a bundle row. ``ray_ids`` identifies the
    transport path, and ``path_order`` determines row order within that path.
    Overlapping particle chords are intentionally unsupported and fail loudly.

    Dirty-path ballistic intensity uses the replacement inclusion model:

    - background intervals attenuate with medium scattering plus absorption (``mu_s + mu_a``)
    - particle intervals attenuate with particle ``mu_total``
    - particle ``I_before`` / ``E_scat`` therefore see upstream background loss

    Optical coefficients should match those used by downstream source deposition.
    If omitted, background intervals do not attenuate the dirty intensity profile
    (particle-only extinction), which is incorrect whenever medium absorption/scatter
    coefficients are nonzero.

    Notebook / protocol:
        Particle transport-pair construction (M5B).
    """
    if not isinstance(segments, RaySegmentBundle):
        raise TypeError("segments must be a RaySegmentBundle with path lineage")

    mu_s_v, mu_a_v, mu_bg = _resolve_background_mu(material, mu_s, mu_a)

    particle_set = _as_particle_set(particles)
    event_list = list(
        intersect_segments_with_particles(
            segments.starts,
            segments.ends,
            particle_set,
            path_length_tol=path_length_tol,
        )
        if events is None
        else events
    )
    _validate_events(segments, particle_set, event_list)

    events_by_segment: dict[int, list[ParticleIntersectionEvent]] = {}
    for event in event_list:
        events_by_segment.setdefault(int(event.segment_index), []).append(event)
    for segment_events in events_by_segment.values():
        segment_events.sort(key=lambda event: (event.entry_t, event.particle_index))
        segment_index = segment_events[0].segment_index
        segment_length = float(
            np.linalg.norm(
                segments.ends[segment_index] - segments.starts[segment_index]
            )
        )
        parameter_tol = path_length_tol / max(segment_length, path_length_tol)
        previous_exit = -np.inf
        for event in segment_events:
            if event.entry_t < previous_exit - parameter_tol:
                raise ValueError(
                    "Overlapping particle intervals are not supported; "
                    f"segment {event.segment_index} has overlapping chords"
                )
            previous_exit = max(previous_exit, event.exit_t)

    affected_rows = sorted(events_by_segment)
    affected_paths = sorted({int(segments.ray_ids[i]) for i in affected_rows})
    pairs: list[AffectedTransportPair] = []
    total_E_abs = 0.0
    total_E_scat = 0.0

    for path_id in affected_paths:
        path_rows = np.flatnonzero(segments.ray_ids == path_id)
        path_rows = path_rows[np.argsort(segments.path_order[path_rows])]
        clean_intervals: list[TransportInterval] = []
        dirty_intervals: list[TransportInterval] = []
        path_events: list[ParticleIntersectionEvent] = []
        scatter_events: list[ParticleScatterSourceEvent] = []
        path_E_abs = 0.0
        path_E_scat = 0.0
        path_particle_factor = 1.0
        path_s = 0.0
        dirty_interval_id = 0
        clean_output = 0.0
        dirty_output = 0.0

        for clean_interval_id, row in enumerate(path_rows.tolist()):
            start = np.asarray(segments.starts[row], dtype=float)
            end = np.asarray(segments.ends[row], dtype=float)
            direction = end - start
            length = float(np.linalg.norm(direction))
            parameter_tol = path_length_tol / max(length, path_length_tol)
            I_clean = float(segments.intensities[row])
            clean_intervals.append(
                TransportInterval(
                    path_id=path_id,
                    segment_index=row,
                    interval_id=clean_interval_id,
                    s0=path_s,
                    s1=path_s + length,
                    start=start,
                    end=end,
                    I_in=I_clean,
                    I_out=I_clean,
                    metadata={"role": "background", "transport": "clean"},
                )
            )

            I_dirty = I_clean * path_particle_factor
            cursor_t = 0.0
            segment_particle_factor = 1.0
            segment_events = events_by_segment.get(row, [])
            path_events.extend(segment_events)

            for event in segment_events:
                if event.entry_t > cursor_t + parameter_tol:
                    bg_start = start + cursor_t * direction
                    bg_end = event.entry_point
                    bg_length = float(np.linalg.norm(bg_end - bg_start))
                    I_in = I_dirty
                    I_out = _background_interval_exit_intensity(I_in, bg_length, mu_bg)
                    dirty_intervals.append(
                        TransportInterval(
                            path_id=path_id,
                            segment_index=row,
                            interval_id=dirty_interval_id,
                            s0=path_s + cursor_t * length,
                            s1=path_s + event.entry_t * length,
                            start=bg_start,
                            end=bg_end,
                            I_in=I_in,
                            I_out=I_out,
                            metadata={"role": "background", "transport": "dirty"},
                        )
                    )
                    dirty_interval_id += 1
                    I_dirty = I_out

                particle = particle_set[event.particle_index]
                I_before = I_dirty
                attenuation = float(
                    np.exp(-particle.mu_total * event.path_length_inside_particle)
                )
                I_after = I_before * attenuation
                E_loss = I_before - I_after
                E_abs, E_scat = partition_particle_loss(
                    E_loss, particle.mu_abs, particle.mu_scat
                )
                particle_interval_id = dirty_interval_id
                dirty_intervals.append(
                    TransportInterval(
                        path_id=path_id,
                        segment_index=row,
                        interval_id=particle_interval_id,
                        s0=path_s + event.entry_t * length,
                        s1=path_s + event.exit_t * length,
                        start=event.entry_point,
                        end=event.exit_point,
                        I_in=I_before,
                        I_out=I_after,
                        active_particle_indices=(event.particle_index,),
                        metadata={"role": "particle", "transport": "dirty"},
                    )
                )
                dirty_interval_id += 1
                scatter_events.append(
                    ParticleScatterSourceEvent(
                        path_id=path_id,
                        segment_index=row,
                        interval_id=particle_interval_id,
                        particle_index=event.particle_index,
                        entry_point=event.entry_point,
                        exit_point=event.exit_point,
                        point=event.midpoint_inside_particle,
                        E_scat=E_scat,
                        E_abs=E_abs,
                        I_before=I_before,
                        I_after=I_after,
                        metadata={
                            "mu_abs": particle.mu_abs,
                            "mu_scat": particle.mu_scat,
                            "path_length": event.path_length_inside_particle,
                        },
                    )
                )
                I_dirty = I_after
                segment_particle_factor *= attenuation
                path_E_abs += E_abs
                path_E_scat += E_scat
                cursor_t = event.exit_t

            if cursor_t < 1.0 - parameter_tol:
                bg_start = start + cursor_t * direction
                bg_end = end
                bg_length = float(np.linalg.norm(bg_end - bg_start))
                I_in = I_dirty
                I_out = _background_interval_exit_intensity(I_in, bg_length, mu_bg)
                dirty_intervals.append(
                    TransportInterval(
                        path_id=path_id,
                        segment_index=row,
                        interval_id=dirty_interval_id,
                        s0=path_s + cursor_t * length,
                        s1=path_s + length,
                        start=bg_start,
                        end=bg_end,
                        I_in=I_in,
                        I_out=I_out,
                        metadata={"role": "background", "transport": "dirty"},
                    )
                )
                dirty_interval_id += 1
                I_dirty = I_out

            path_particle_factor *= segment_particle_factor
            path_s += length
            clean_output = I_clean
            dirty_output = I_dirty

        total_E_abs += path_E_abs
        total_E_scat += path_E_scat
        pairs.append(
            AffectedTransportPair(
                path_id=path_id,
                clean_intervals=tuple(clean_intervals),
                dirty_intervals=tuple(dirty_intervals),
                particle_events=tuple(path_events),
                particle_scatter_source_events=tuple(scatter_events),
                clean_output_intensity=clean_output,
                dirty_output_intensity=dirty_output,
                total_E_abs=path_E_abs,
                total_E_scat=path_E_scat,
                metadata={
                    "partition_model": PARTITION_MODEL,
                    "replacement_inclusion": True,
                    "background_mu_s": mu_s_v,
                    "background_mu_a": mu_a_v,
                },
            )
        )

    return AffectedTransportPairResult(
        pairs=tuple(pairs),
        affected_path_ids=tuple(affected_paths),
        affected_segment_indices=np.asarray(affected_rows, dtype=int),
        total_E_abs=float(total_E_abs),
        total_E_scat=float(total_E_scat),
        metadata={
            "n_pairs": len(pairs),
            "n_particle_events": len(event_list),
            "path_identity_field": "RaySegmentBundle.ray_ids",
            "path_order_field": "RaySegmentBundle.path_order",
            "overlap_composition": "unsupported",
            "requires_remeshing": False,
            "changes_diffusion_operator": False,
            "background_mu_s": mu_s_v,
            "background_mu_a": mu_a_v,
        },
    )


def _intervals_to_bundle(
    intervals: Sequence[TransportInterval],
    *,
    background_only: bool,
) -> RaySegmentBundle:
    selected = [
        interval
        for interval in intervals
        if (not background_only or not interval.active_particle_indices)
        and interval.length > DEFAULT_PATH_LENGTH_TOL
    ]
    if not selected:
        return RaySegmentBundle(
            starts=np.zeros((0, 3), dtype=float),
            ends=np.zeros((0, 3), dtype=float),
            intensities=np.zeros(0, dtype=float),
            ray_ids=np.zeros(0, dtype=int),
        )

    path_order = np.empty(len(selected), dtype=int)
    next_order: dict[int, int] = {}
    for i, interval in enumerate(selected):
        order = next_order.get(interval.path_id, 0)
        path_order[i] = order
        next_order[interval.path_id] = order + 1
    return RaySegmentBundle(
        starts=np.asarray([interval.start for interval in selected], dtype=float),
        ends=np.asarray([interval.end for interval in selected], dtype=float),
        intensities=np.asarray([interval.I_in for interval in selected], dtype=float),
        ray_ids=np.asarray([interval.path_id for interval in selected], dtype=int),
        segment_ids=np.arange(len(selected), dtype=int),
        path_order=path_order,
    )


def _barycentric_tet_coords(
    point: np.ndarray,
    tet_vertices: np.ndarray,
) -> np.ndarray | None:
    v = np.asarray(tet_vertices, dtype=float)
    p = np.asarray(point, dtype=float).reshape(3)
    mat = np.column_stack((v[1] - v[0], v[2] - v[0], v[3] - v[0]))
    try:
        b123 = np.linalg.solve(mat, p - v[0])
    except np.linalg.LinAlgError:
        return None
    return np.asarray(
        [1.0 - float(np.sum(b123)), b123[0], b123[1], b123[2]], dtype=float
    )


def point_in_tetrahedron(
    point: np.ndarray,
    tet_vertices: np.ndarray,
    *,
    eps: float = 1e-10,
) -> bool:
    """Return True when ``point`` lies inside the tetrahedron within ``eps``."""
    coords = _barycentric_tet_coords(point, tet_vertices)
    return bool(
        coords is not None
        and np.all(coords >= -eps)
        and np.all(coords <= 1.0 + eps)
    )


def find_containing_tet(
    diffusion_mesh: DiffusionMesh,
    point: np.ndarray,
    *,
    eps: float = 1e-10,
) -> int | None:
    """Return the index of the tet containing ``point``, or ``None`` if outside.

    Args:
        diffusion_mesh: Coarse diffusion mesh with node and tet tables.
        point: Query location ``(3,)``.
        eps: Barycentric tolerance for boundary inclusion.

    Returns:
        Tet index or ``None`` when no tet contains the point.
    """
    nodes = np.asarray(diffusion_mesh.nodes, dtype=float)
    for tet_index, tet in enumerate(np.asarray(diffusion_mesh.tets, dtype=int)):
        if point_in_tetrahedron(point, nodes[tet], eps=eps):
            return int(tet_index)
    return None


def nearest_tet_centroid(
    diffusion_mesh: DiffusionMesh,
    point: np.ndarray,
) -> int:
    """Return the tet index whose centroid is nearest ``point``."""
    centroids = np.asarray(diffusion_mesh.centroids, dtype=float)
    p = np.asarray(point, dtype=float).reshape(3)
    return int(np.argmin(np.sum((centroids - p) ** 2, axis=1)))


def _deposit_event_at_point(
    diffusion_mesh: DiffusionMesh,
    point: np.ndarray,
    *,
    assignment: Literal["containing_tet", "nearest_centroid"],
    outside_policy: Literal["error", "nearest_centroid"],
) -> tuple[int, str]:
    """Assign integrated energy to one tet (explicit midpoint fallback path)."""
    if assignment == "nearest_centroid":
        return nearest_tet_centroid(diffusion_mesh, point), "nearest_centroid"
    if assignment == "containing_tet":
        tet_index = find_containing_tet(diffusion_mesh, point)
        if tet_index is not None:
            return tet_index, "containing_tet"
        if outside_policy == "error":
            raise ValueError("Particle scatter point lies outside the diffusion mesh")
        return nearest_tet_centroid(diffusion_mesh, point), "nearest_centroid"
    raise ValueError(f"Unknown point assignment mode: {assignment!r}")


def _event_particle_optical_coeffs(
    event: ParticleScatterSourceEvent,
) -> tuple[float, float]:
    """Return absorption and scatter coefficients ``(mu_abs, mu_scat)`` used for this dirty particle interval."""
    mu_abs = float(event.metadata.get("mu_abs", 0.0))
    mu_scat = float(event.metadata.get("mu_scat", 0.0))
    if mu_abs < 0.0 or mu_scat < 0.0:
        raise ValueError("Particle scatter-event optical coefficients must be >= 0")
    return mu_abs, mu_scat


def _deposit_event_attenuated_chord(
    diffusion_mesh: DiffusionMesh,
    event: ParticleScatterSourceEvent,
) -> np.ndarray:
    """Deposit particle scatter with local Beer–Lambert intensity along the chord.

    Uses the dirty intensity at particle entry (``I_before``) and the particle
    coefficients so that local scatter is highest near entry and decays as the
    ray attenuates inside the particle — the same exact ray–tet machinery as
    background ``deposit_ray_source``.
    """
    mu_abs, mu_scat = _event_particle_optical_coeffs(event)
    if mu_scat <= 0.0 or event.I_before <= 0.0:
        return np.zeros(diffusion_mesh.n_tets, dtype=float)

    chord = RaySegmentBundle(
        starts=np.asarray(event.entry_point, dtype=float).reshape(1, 3),
        ends=np.asarray(event.exit_point, dtype=float).reshape(1, 3),
        intensities=np.asarray([event.I_before], dtype=float),
        ray_ids=np.asarray([event.path_id], dtype=int),
    )
    deposition = deposit_ray_source(
        diffusion_mesh,
        chord,
        mu_s=mu_scat,
        mu_a=mu_abs,
    )
    return np.asarray(deposition.E_scat_elem, dtype=float)


def deposit_particle_scatter_sources(
    diffusion_mesh: DiffusionMesh,
    pair_result: AffectedTransportPairResult,
    *,
    assignment: Literal[
        "attenuated_chord",
        "chord_length",
        "containing_tet",
        "nearest_centroid",
    ] = "attenuated_chord",
    outside_policy: Literal["error", "nearest_centroid"] = "nearest_centroid",
) -> ParticleScatterDepositionResult:
    """Deposit scatter events owned by dirty transport pairs.

    Default ``assignment="attenuated_chord"`` deposits each event along the
    inside-particle chord with local Beer–Lambert intensity starting from
    ``I_before`` and the particle ``mu_abs`` / ``mu_scat``.  Scatter is therefore
    highest near particle entry and can fall below the replaced background only
    after sufficient in-particle attenuation.

    ``chord_length`` keeps the older uniform path-length split of the integrated
    ``E_scat``.  ``containing_tet`` / ``nearest_centroid`` remain explicit
    single-element midpoint fallbacks.
    """
    delta = np.zeros(diffusion_mesh.n_tets, dtype=float)
    events = [
        event
        for pair in pair_result.pairs
        for event in pair.particle_scatter_source_events
        if event.E_scat > 0.0
    ]
    assignment_counts = {
        "attenuated_chord": 0,
        "chord_length": 0,
        "containing_tet": 0,
        "nearest_centroid": 0,
    }

    for event in events:
        if assignment == "attenuated_chord":
            chord_delta = _deposit_event_attenuated_chord(diffusion_mesh, event)
            deposited = float(np.sum(chord_delta))
            if deposited > 0.0:
                delta += chord_delta
                assignment_counts["attenuated_chord"] += 1
                continue
            if outside_policy == "error":
                raise ValueError(
                    "Particle scatter chord does not intersect the diffusion mesh"
                )
            tet_index = nearest_tet_centroid(diffusion_mesh, event.point)
            delta[tet_index] += event.E_scat
            assignment_counts["nearest_centroid"] += 1
            continue

        if assignment == "chord_length":
            chord_delta = distribute_energy_along_segment(
                diffusion_mesh,
                event.entry_point,
                event.exit_point,
                event.E_scat,
            )
            deposited = float(np.sum(chord_delta))
            if deposited > 0.0:
                delta += chord_delta
                assignment_counts["chord_length"] += 1
                continue
            if outside_policy == "error":
                raise ValueError(
                    "Particle scatter chord does not intersect the diffusion mesh"
                )
            tet_index = nearest_tet_centroid(diffusion_mesh, event.point)
            delta[tet_index] += event.E_scat
            assignment_counts["nearest_centroid"] += 1
            continue

        if assignment in ("containing_tet", "nearest_centroid"):
            tet_index, used = _deposit_event_at_point(
                diffusion_mesh,
                event.point,
                assignment=assignment,
                outside_policy=outside_policy,
            )
            delta[tet_index] += event.E_scat
            assignment_counts[used] += 1
            continue

        raise ValueError(f"Unknown assignment mode: {assignment!r}")

    total_events = float(sum(event.E_scat for event in events))
    if assignment == "attenuated_chord":
        distribution = "exact_ray_tet_beer_lambert"
    elif assignment == "chord_length":
        distribution = "exact_ray_tet_chord_length"
    else:
        distribution = "midpoint_point_assignment"

    return ParticleScatterDepositionResult(
        delta_E_particle_scat_elem=delta,
        assignment_mode=assignment,
        outside_policy=outside_policy,
        n_events=len(events),
        n_assigned=len(events),
        total_E_scat_events=total_events,
        total_E_scat_deposited=float(np.sum(delta)),
        metadata={
            "source": "dirty_transport_pair_events",
            "assignment_counts": assignment_counts,
            "distribution": distribution,
        },
    )


def compute_transport_source_correction(
    diffusion_mesh: DiffusionMesh,
    pair_result: AffectedTransportPairResult,
    E_clean_elem: np.ndarray,
    material: OpticalMaterialConfig | None = None,
    *,
    mu_s: float | None = None,
    mu_a: float | None = None,
    assignment: Literal[
        "attenuated_chord",
        "chord_length",
        "containing_tet",
        "nearest_centroid",
    ] = "attenuated_chord",
    outside_policy: Literal["error", "nearest_centroid"] = "nearest_centroid",
) -> TransportSourceCorrectionResult:
    """Compute diffusion source correction from dirty-minus-clean path contributions.

    Forms ``delta_E_transport_elem = delta_E_background_elem + delta_E_particle_scat_elem``
    and derives ``S_particle`` for the particle diffusion solve.

    Notebook / protocol:
        Particle source correction (M5C).
    """
    E_clean = np.asarray(E_clean_elem, dtype=float)
    if E_clean.shape != (diffusion_mesh.n_tets,):
        raise ValueError(
            f"E_clean_elem must have shape ({diffusion_mesh.n_tets},)"
        )

    clean_intervals = [
        interval for pair in pair_result.pairs for interval in pair.clean_intervals
    ]
    dirty_intervals = [
        interval for pair in pair_result.pairs for interval in pair.dirty_intervals
    ]
    clean_bundle = _intervals_to_bundle(clean_intervals, background_only=False)
    dirty_bundle = _intervals_to_bundle(dirty_intervals, background_only=True)
    clean_dep = deposit_ray_source(
        diffusion_mesh, clean_bundle, material, mu_s=mu_s, mu_a=mu_a
    )
    dirty_dep = deposit_ray_source(
        diffusion_mesh, dirty_bundle, material, mu_s=mu_s, mu_a=mu_a
    )
    delta_background = (
        np.asarray(dirty_dep.E_scat_elem, dtype=float)
        - np.asarray(clean_dep.E_scat_elem, dtype=float)
    )
    particle_scatter = deposit_particle_scatter_sources(
        diffusion_mesh,
        pair_result,
        assignment=assignment,
        outside_policy=outside_policy,
    )
    delta_particle_scat = particle_scatter.delta_E_particle_scat_elem
    delta_transport = delta_background + delta_particle_scat
    E_particle = E_clean + delta_transport
    S_particle = np.zeros_like(E_particle)
    np.divide(
        E_particle,
        np.asarray(diffusion_mesh.volumes, dtype=float),
        out=S_particle,
        where=np.asarray(diffusion_mesh.volumes, dtype=float) > 0.0,
    )

    return TransportSourceCorrectionResult(
        E_clean_elem=E_clean,
        delta_E_background_elem=delta_background,
        delta_E_particle_scat_elem=delta_particle_scat,
        delta_E_transport_elem=delta_transport,
        E_particle_elem=E_particle,
        S_particle=S_particle,
        clean_affected_deposition=clean_dep,
        dirty_affected_deposition=dirty_dep,
        particle_scatter_deposition=particle_scatter,
        metadata={
            "source_model": "affected_transport_pair_delta",
            "n_affected_pairs": len(pair_result.pairs),
            "requires_remeshing": False,
            "changes_diffusion_operator": False,
        },
    )


@dataclass(frozen=True)
class TransportPairDepositionResult:
    """Per-path clean, dirty-background, and particle-scatter deposition.

    Attributes:
        E_clean_elem: Clean background scatter on affected intervals.
        E_dirty_background_elem: Dirty background scatter on affected intervals.
        E_particle_scat_elem: Particle scatter deposition for the path.
        E_dirty_total_elem: Sum of dirty background and particle scatter.
        particle_scatter_deposition: Detailed scatter deposition result.
        metadata: Path id and assignment mode.
    """

    E_clean_elem: np.ndarray
    E_dirty_background_elem: np.ndarray
    E_particle_scat_elem: np.ndarray
    E_dirty_total_elem: np.ndarray
    particle_scatter_deposition: ParticleScatterDepositionResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "E_clean_elem",
            "E_dirty_background_elem",
            "E_particle_scat_elem",
            "E_dirty_total_elem",
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        object.__setattr__(self, "metadata", dict(self.metadata))


def deposit_transport_pair_sources(
    diffusion_mesh: DiffusionMesh,
    pair: AffectedTransportPair,
    material: OpticalMaterialConfig | None = None,
    *,
    mu_s: float | None = None,
    mu_a: float | None = None,
    assignment: Literal[
        "attenuated_chord",
        "chord_length",
        "containing_tet",
        "nearest_centroid",
    ] = "attenuated_chord",
    outside_policy: Literal["error", "nearest_centroid"] = "nearest_centroid",
) -> TransportPairDepositionResult:
    """Deposit one affected pair with the same routines as batch source correction."""
    clean_bundle = _intervals_to_bundle(pair.clean_intervals, background_only=False)
    dirty_bundle = _intervals_to_bundle(pair.dirty_intervals, background_only=True)
    clean_dep = deposit_ray_source(
        diffusion_mesh, clean_bundle, material, mu_s=mu_s, mu_a=mu_a
    )
    dirty_dep = deposit_ray_source(
        diffusion_mesh, dirty_bundle, material, mu_s=mu_s, mu_a=mu_a
    )
    pair_result = AffectedTransportPairResult(
        pairs=(pair,),
        affected_path_ids=(pair.path_id,),
        affected_segment_indices=np.asarray([], dtype=int),
        total_E_abs=float(pair.total_E_abs),
        total_E_scat=float(pair.total_E_scat),
    )
    particle_scatter = deposit_particle_scatter_sources(
        diffusion_mesh,
        pair_result,
        assignment=assignment,
        outside_policy=outside_policy,
    )
    E_clean = np.asarray(clean_dep.E_scat_elem, dtype=float)
    E_dirty_bg = np.asarray(dirty_dep.E_scat_elem, dtype=float)
    E_particle = np.asarray(particle_scatter.delta_E_particle_scat_elem, dtype=float)
    return TransportPairDepositionResult(
        E_clean_elem=E_clean,
        E_dirty_background_elem=E_dirty_bg,
        E_particle_scat_elem=E_particle,
        E_dirty_total_elem=E_dirty_bg + E_particle,
        particle_scatter_deposition=particle_scatter,
        metadata={
            "path_id": pair.path_id,
            "assignment_mode": particle_scatter.assignment_mode,
        },
    )


def assert_downstream_background_shadow(
    diffusion_mesh: DiffusionMesh,
    pair: AffectedTransportPair,
    material: OpticalMaterialConfig | None = None,
    *,
    mu_s: float | None = None,
    mu_a: float | None = None,
    density: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-15,
) -> dict[str, Any]:
    """Assert dirty background deposition does not exceed clean downstream.

    Downstream is defined by centroid projection beyond the last particle-exit
    coordinate along the clean path direction.  Compares background deposition
    only (particle-scatter source is excluded).
    """
    deposition = deposit_transport_pair_sources(
        diffusion_mesh, pair, material, mu_s=mu_s, mu_a=mu_a
    )
    volumes = np.asarray(diffusion_mesh.volumes, dtype=float)
    if density:
        clean_values = deposition.E_clean_elem / volumes
        dirty_values = deposition.E_dirty_background_elem / volumes
        value_label = "source density"
    else:
        clean_values = deposition.E_clean_elem
        dirty_values = deposition.E_dirty_background_elem
        value_label = "energy"

    particle_intervals = [
        interval
        for interval in pair.dirty_intervals
        if interval.active_particle_indices
    ]
    if not particle_intervals:
        raise AssertionError(f"path {pair.path_id}: no particle intervals found")

    if not pair.clean_intervals:
        raise AssertionError(f"path {pair.path_id}: no clean intervals found")

    p_start = np.asarray(pair.clean_intervals[0].start, dtype=float)
    p_end = np.asarray(pair.clean_intervals[-1].end, dtype=float)
    direction = p_end - p_start
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        direction = direction / norm

    elem_indices = np.asarray(
        sorted(
            set(np.nonzero(clean_values)[0].tolist())
            | set(np.nonzero(dirty_values)[0].tolist())
        ),
        dtype=int,
    )
    if len(elem_indices) == 0:
        raise AssertionError(f"path {pair.path_id}: no deposited elements found")

    centroids = np.asarray(diffusion_mesh.centroids, dtype=float)
    elem_s = centroids[elem_indices] @ direction - p_start @ direction
    order = np.argsort(elem_s)
    elem_indices = elem_indices[order]
    elem_s = elem_s[order]

    downstream_start_s = max(
        float(np.asarray(interval.end, dtype=float) @ direction - p_start @ direction)
        for interval in particle_intervals
    )
    downstream_mask = elem_s > downstream_start_s

    violations: list[dict[str, float | int]] = []
    for elem, s, clean_val, dirty_val, is_downstream in zip(
        elem_indices,
        elem_s,
        clean_values[elem_indices],
        dirty_values[elem_indices],
        downstream_mask,
    ):
        if not is_downstream:
            continue
        allowed = clean_val * (1.0 + rtol) + atol
        if dirty_val > allowed:
            violations.append(
                {
                    "elem": int(elem),
                    "s": float(s),
                    "clean": float(clean_val),
                    "dirty_background": float(dirty_val),
                    "excess": float(dirty_val - clean_val),
                    "ratio": float(dirty_val / clean_val) if clean_val > 0 else np.inf,
                }
            )

    summary = {
        "path_id": pair.path_id,
        "quantity": value_label,
        "downstream_start_s": downstream_start_s,
        "n_downstream_elements": int(np.count_nonzero(downstream_mask)),
        "n_violations": len(violations),
        "violations": violations,
        "deposition": deposition,
    }
    if violations:
        raise AssertionError(
            f"path {pair.path_id}: dirty background deposition exceeds clean "
            f"background deposition downstream of the particle "
            f"({len(violations)} element(s))"
        )
    return summary

