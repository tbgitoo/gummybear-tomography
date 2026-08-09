"""Serial multi-view sequence execution with persistent source caches.

Default backend: build clean source once, particle source once, solve diffusion
once per role, then capture ordered camera views. Views within a sequence stay
serial; independent sequences may run in parallel at the plan layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from inspect import signature
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
import warnings

import numpy as np

from gummybear.datasets.cache_keys import (
    CAMERA_LOOK_AT_POLICY,
    DEFAULT_BARYCENTRIC_TOLERANCE,
    camera_visibility_cache_key,
    camera_visibility_cache_key_payload,
    phi_sampling_localization_cache_key,
    phi_sampling_localization_cache_key_payload,
)
from gummybear.datasets.generation_plan import (
    CameraPose,
    SequenceJob,
    resolve_job_output_identity,
)
from gummybear.datasets.manifest_writer import build_sequence_manifest
from gummybear.datasets.output_plan import (
    OUTPUT_COMPLETE_CURRENT,
    OUTPUT_DISABLED_NOT_RUN,
    OUTPUT_MISSING,
    OUTPUT_ORPHANED_NOT_REQUESTED,
    OutputDeltaItem,
    resolve_output_root,
)
from gummybear.datasets.sequence_writer import write_sequence_roles
from gummybear.datasets.source_cache import (
    CAMERA_VISIBILITY_ALLOW_NONFINITE_ARRAYS,
    CAMERA_VISIBILITY_PAYLOAD_SCHEMA_VERSION,
    CAMERA_VISIBILITY_REQUIRED_ARRAYS,
    CLEAN_ALLOW_NONFINITE_ARRAYS,
    CLEAN_PAYLOAD_SCHEMA_VERSION,
    CLEAN_REQUIRED_ARRAYS,
    PARTICLE_PAYLOAD_SCHEMA_VERSION,
    PARTICLE_REQUIRED_ARRAYS,
    PHI_SAMPLING_LOCALIZATION_PAYLOAD_SCHEMA_VERSION,
    PHI_SAMPLING_LOCALIZATION_REQUIRED_ARRAYS,
    CacheEvent,
    SourceCacheStore,
)
from gummybear.datasets.text_table import format_text_table, short_hash
from gummybear.geometry import load_stl
from gummybear.paths import display_path
from gummybear.optics import (
    PointLightConfig,
    PhiSamplingLocalization,
    RaySegmentBundle,
    RefractedRayBundleResult,
    SourceDepositionResult,
    SourceSamplingParams,
    compose_hybrid_image,
    compute_refractive_direct_image,
    deposit_ray_source,
    generate_diffusion_mesh,
    in_object_segments_from_rays,
    localize_points_in_diffusion_mesh,
    make_source_ray_bundle,
    refract_ray_bundle,
    sample_diffuse_image,
    solve_diffusion,
)
from gummybear.particles import (
    ParticleSet,
    ParticleSphere,
    build_affected_transport_pairs,
    compute_transport_source_correction,
)
from gummybear.rays import (
    PinholeCameraConfig,
    SourceRayBundle,
    first_visible_hits_with_points,
    make_camera_rays,
)


@dataclass(frozen=True)
class SmokeRuntimeSettings:
    """Runtime knobs recorded in manifests but not in workbook columns or cache keys.

    Attributes:
        target_elements: Coarse diffusion tet count (fixed at 1000 for cache identity).
        exitance_scale: Diffuse camera sampling scale.
        camera_fov_deg: Pinhole field of view in degrees.
        n_from: Exterior refractive index for source refraction.
        source_delta_assignment: Particle scatter deposition mode (``attenuated_chord``).
        surface_interpolation: Fluence field ``Phi`` sampling method at camera hits.
    """

    target_elements: int = 1000
    exitance_scale: float = 10.0
    camera_fov_deg: float = 35.0
    n_from: float = 1.0
    source_delta_assignment: str = "attenuated_chord"
    surface_interpolation: str = "tetrahedral_barycentric"


@dataclass(frozen=True)
class CapturedFrame:
    """One captured clean/particle camera-intensity frame before role encoding.

    Attributes:
        clean: Linear clean-role intensity grid ``[H, W]``.
        particle: Linear particle-role intensity grid ``[H, W]``.
        metadata: Pose, cache events, and camera parameters for the manifest.
    """
    clean: np.ndarray
    particle: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GeneratedSequenceResult:
    """Outcome of generating one sequence directory.

    Attributes:
        sequence_id: Generated sequence identifier.
        output_path: Published sequence folder path.
        frame_count: Number of multi-view frames written.
        clean_optical_cache_id, particle_source_cache_id: Source cache keys used.
        stage_seconds: Named timing breakdown for profiling.
        clean_cache, particle_cache: Primary source cache events.
        visibility_cache, phi_localization_cache: Optional per-pose aggregate events.
        warnings: Non-fatal generation notices (e.g. no transport intersection).
    """
    sequence_id: str
    output_path: str
    frame_count: int
    clean_optical_cache_id: str
    particle_source_cache_id: str
    stage_seconds: dict[str, float]
    clean_cache: CacheEvent
    particle_cache: CacheEvent
    visibility_cache: CacheEvent | None = None
    phi_localization_cache: CacheEvent | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationRunResult:
    """Batch outcome from :func:`gummybear.datasets.generation_plan.run_generation_plan`.

    Attributes:
        generated: Successfully generated sequences in this run.
        skipped: Sequence ids already complete on disk.
        failed: Sequence ids that failed (reserved for future batch reporting).
        dry_run: True when physics was not executed.
        output_items: Output reconciliation rows from the delta plan.
        verbose: When True, :meth:`__repr__` keeps the wide diagnostic table;
            when False (default), print one line per job as ``cache`` /
            ``computed`` (plus orphans / failures).
    """
    generated: tuple[GeneratedSequenceResult, ...]
    skipped: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    dry_run: bool = False
    output_items: tuple[OutputDeltaItem, ...] = ()
    verbose: bool = False

    @property
    def warnings(self) -> tuple[str, ...]:
        """Flatten per-sequence warnings for notebook / CLI inspection."""
        return tuple(
            f"{item.sequence_id}: {message}"
            for item in self.generated
            for message in item.warnings
        )

    @property
    def disabled_sequence_ids(self) -> tuple[str, ...]:
        """Sequence ids marked disabled in ``output_items`` (stable order)."""
        return tuple(
            item.sequence_id
            for item in self.output_items
            if item.status == OUTPUT_DISABLED_NOT_RUN
        )

    def _summary_header_lines(self) -> list[str]:
        """Shared counts for compact and verbose ``repr`` summaries."""
        n_disabled = len(self.disabled_sequence_ids)
        lines = [
            "GenerationRunResult(",
            f"  dry_run={self.dry_run!r},",
            f"  generated={len(self.generated)},",
            f"  skipped={len(self.skipped)},",
            f"  failed={len(self.failed)},",
            f"  disabled={n_disabled},",
            f"  warnings={len(self.warnings)},",
        ]
        if n_disabled == 0:
            lines.append("  disabled_rows=(none),")
        return lines

    def _disposition(
        self,
        sequence_id: str,
        *,
        status: str | None = None,
    ) -> str:
        """Compact per-job label for quiet summaries."""
        if status == OUTPUT_ORPHANED_NOT_REQUESTED:
            return "orphaned"
        if status == OUTPUT_DISABLED_NOT_RUN:
            return "disabled"
        if sequence_id in set(self.failed):
            return "failed"
        if sequence_id in {item.sequence_id for item in self.generated}:
            return "computed"
        if sequence_id in set(self.skipped):
            if self.dry_run and status == OUTPUT_MISSING:
                return "would_compute"
            return "cache"
        if self.dry_run and status == OUTPUT_MISSING:
            return "would_compute"
        if status == OUTPUT_COMPLETE_CURRENT:
            return "cache"
        return status.removeprefix("output_") if status else "-"

    def _iter_summary_entries(self) -> list[tuple[str, str, str, str]]:
        """Return ``(sequence_id, status, reason, output_path)`` rows in display order."""
        generated_by_id = {item.sequence_id: item for item in self.generated}
        seen: set[str] = set()
        entries: list[tuple[str, str, str, str]] = []

        def _append(
            *,
            sequence_id: str,
            status: str,
            reason: str,
            output_path: str,
        ) -> None:
            if sequence_id in seen:
                return
            seen.add(sequence_id)
            entries.append((sequence_id, status, reason, output_path))

        for item in self.output_items:
            _append(
                sequence_id=item.sequence_id,
                status=item.status,
                reason=item.reason,
                output_path=item.output_path,
            )
        for generated in self.generated:
            _append(
                sequence_id=generated.sequence_id,
                status="generated",
                reason="generated",
                output_path=generated.output_path,
            )
        for sequence_id in self.skipped:
            _append(
                sequence_id=sequence_id,
                status="skipped",
                reason="skipped",
                output_path="",
            )
        for sequence_id in self.failed:
            _append(
                sequence_id=sequence_id,
                status="failed",
                reason="failed",
                output_path="",
            )
        return entries

    def _format_compact(self) -> str:
        """One line per job: id + cache/computed (orphans always included)."""
        lines = self._summary_header_lines()
        lines.append("  jobs=")
        entries = self._iter_summary_entries()
        if not entries:
            lines.append("    (empty)")
        else:
            width = max(len(sequence_id) for sequence_id, _, _, _ in entries)
            for sequence_id, status, _reason, _path in entries:
                disposition = self._disposition(sequence_id, status=status)
                lines.append(f"    {sequence_id:<{width}}  {disposition}")
        if self.warnings:
            lines.append("  warning_messages=")
            for message in self.warnings:
                lines.append(f"    - {message}")
        lines.append(")")
        return "\n".join(lines)

    def _format_verbose(self) -> str:
        """Wide diagnostic table with cache ids (legacy notebook output)."""
        generated_by_id = {item.sequence_id: item for item in self.generated}
        skipped_ids = set(self.skipped)
        failed_ids = set(self.failed)

        headers = (
            "sequence_id",
            "run",
            "status",
            "reason",
            "frames",
            "clean",
            "particle",
            "secs",
            "output_path",
        )
        rows: list[tuple[str, ...]] = []

        def _run_label(sequence_id: str, *, status: str | None = None) -> str:
            if sequence_id in failed_ids:
                return "failed"
            if sequence_id in generated_by_id:
                return "generated"
            if sequence_id in skipped_ids:
                return "skipped"
            if status == OUTPUT_DISABLED_NOT_RUN:
                return "disabled"
            if status == OUTPUT_ORPHANED_NOT_REQUESTED:
                return "orphaned"
            if self.dry_run:
                return "dry"
            return "-"

        for sequence_id, status, reason, output_path in self._iter_summary_entries():
            generated = generated_by_id.get(sequence_id)
            if generated is not None:
                frames = str(generated.frame_count)
                clean = generated.clean_cache.status
                particle = generated.particle_cache.status
                total_secs = sum(generated.stage_seconds.values())
                secs = f"{total_secs:.1f}"
                path = display_path(generated.output_path)
            else:
                frames = clean = particle = secs = "-"
                path = display_path(output_path) if output_path else "-"
            rows.append(
                (
                    sequence_id,
                    _run_label(sequence_id, status=status),
                    status.removeprefix("output_"),
                    reason or "-",
                    frames,
                    clean,
                    particle,
                    secs,
                    path,
                )
            )

        lines = self._summary_header_lines()
        if rows:
            lines.append("  sequences=")
            table = format_text_table(headers, rows)
            lines.extend(f"    {line}" for line in table.splitlines())
        else:
            lines.append("  sequences=(empty),")

        if self.generated:
            lines.append("  cache_ids=")
            for generated in self.generated:
                lines.append(
                    "    "
                    f"{generated.sequence_id}: "
                    f"clean={short_hash(generated.clean_optical_cache_id)}, "
                    f"particle={short_hash(generated.particle_source_cache_id)}"
                )
        if self.warnings:
            lines.append("  warning_messages=")
            for message in self.warnings:
                lines.append(f"    - {message}")
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Readable summary with repo-relative paths (no absolute home paths)."""
        if self.verbose:
            return self._format_verbose()
        return self._format_compact()


class SmokePhysicsBackend(Protocol):
    """Injection seam for serial sequence orchestration tests.

    Implementations must prepare clean and particle source state, solve fields
    once, then capture each camera pose. Optional serialize/restore hooks enable
    persistent source caches.
    """

    def prepare_clean(
        self,
        job: SequenceJob,
        settings: SmokeRuntimeSettings,
    ) -> Any:
        """Build clean optical source state for ``job`` (mesh, rays, deposition).

        Returns:
            Opaque clean state consumed by ``prepare_particle``, ``solve_fields``,
            and ``capture_frame``.
        """
        ...

    def prepare_particle(
        self,
        job: SequenceJob,
        clean_state: Any,
        settings: SmokeRuntimeSettings,
    ) -> Any:
        """Build particle source correction from ``clean_state`` and job particles.

        Returns:
            Opaque particle state with corrected volumetric source density ``S_particle``.
        """
        ...

    def solve_fields(
        self,
        job: SequenceJob,
        clean_state: Any,
        particle_state: Any,
        settings: SmokeRuntimeSettings,
    ) -> Any:
        """Solve clean and particle fluence field ``Phi`` once per sequence via finite-element method (FEM) diffusion.

        Returns:
            Opaque field state holding nodal fluence field ``Phi_nodes`` for both roles.
        """
        ...

    def capture_frame(
        self,
        job: SequenceJob,
        pose: CameraPose,
        clean_state: Any,
        particle_state: Any,
        field_state: Any,
        settings: SmokeRuntimeSettings,
        *,
        visibility_cache_store: SourceCacheStore | None = None,
        force_recompute_visibility: bool = False,
    ) -> CapturedFrame:
        """Render one camera pose from precomputed clean and particle fields.

        Args:
            visibility_cache_store: Optional store for per-pose visibility caches.
            force_recompute_visibility: Rebuild visibility even when cached.

        Returns:
            Linear clean and particle intensity grids for the pose.
        """
        ...

    def diagnostics(
        self,
        clean_state: Any,
        particle_state: Any,
        field_state: Any,
    ) -> dict[str, Any]:
        """Return physics summary metrics for manifest recording."""
        ...


@dataclass
class _CleanState:
    surface_mesh: Any
    diff_mesh: Any
    material: Any
    light: Any
    source_rays: Any
    refracted: Any
    segments: Any
    deposition: Any


@dataclass
class _CleanContext:
    surface_mesh: Any
    diff_mesh: Any
    material: Any
    light: Any


@dataclass
class _ParticleState:
    particles: ParticleSet
    pair_result: Any
    source_correction: Any
    particle_ray_weights: np.ndarray | None = None
    no_transport_intersection: bool = False
    notes: tuple[str, ...] = ()


@dataclass
class _FieldState:
    clean_solve: Any
    particle_solve: Any


@dataclass(frozen=True)
class _CachedPairSummary:
    affected_path_ids: tuple[int, ...]
    affected_segment_indices: np.ndarray
    pairs: tuple[Any, ...] = ()


@dataclass(frozen=True)
class _CachedParticleScatterSummary:
    assignment_mode: str


@dataclass(frozen=True)
class _CachedSourceCorrection:
    E_clean_elem: np.ndarray
    delta_E_background_elem: np.ndarray
    delta_E_particle_scat_elem: np.ndarray
    delta_E_transport_elem: np.ndarray
    E_particle_elem: np.ndarray
    S_particle: np.ndarray
    particle_scatter_deposition: _CachedParticleScatterSummary


def _resolve_job_path(job: SequenceJob) -> Path:
    """Resolve a portable STL triangle mesh file path relative to the workbook or repository root.

    Temporary workbooks outside ``configs/`` are supported by walking upward
    from the workbook location until the relative STL path exists.
    """
    path = Path(job.stl_path)
    if path.is_absolute():
        return path

    candidates: list[Path] = [Path.cwd() / path]
    workbook_path = Path(job.workbook_path).resolve()
    for parent in (workbook_path.parent, *workbook_path.parents):
        candidates.append(parent / path)
        if parent.name == "configs":
            candidates.append(parent.parent / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


def _sample_shape_array(sample_shape: tuple[int, ...] | None) -> np.ndarray:
    if sample_shape is None:
        return np.empty(0, dtype=np.int64)
    return np.asarray(sample_shape, dtype=np.int64)


def _sample_shape_from_array(value: np.ndarray) -> tuple[int, ...] | None:
    shape = tuple(int(item) for item in np.asarray(value, dtype=int).tolist())
    return shape or None


def _mesh_identity(diff_mesh: Any) -> dict[str, Any]:
    return {
        "content_hash": str(diff_mesh.content_hash()),
        "num_nodes": int(diff_mesh.n_nodes),
        "num_tets": int(diff_mesh.n_tets),
    }


def _camera_visibility_surface_identity(job: SequenceJob) -> dict[str, Any]:
    return {
        "identity_kind": "camera_visibility_surface",
        "look_at_policy": CAMERA_LOOK_AT_POLICY,
        "stl_sha256": str(job.stl_sha256),
    }


def _camera_visibility_key_parts(
    job: SequenceJob,
    pose: CameraPose,
    settings: SmokeRuntimeSettings,
) -> tuple[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "stl_sha256": job.stl_sha256 or None,
        "stl_path": None if job.stl_sha256 else job.stl_path,
        "camera_kind": pose.camera_kind,
        "fov_deg": settings.camera_fov_deg,
        "resolution_x": pose.resolution_x,
        "resolution_y": pose.resolution_y,
        "frame_index": pose.frame_index,
        "angle_deg": pose.angle_deg,
        "axis_x": pose.axis[0],
        "axis_y": pose.axis[1],
        "axis_z": pose.axis[2],
        "distance": pose.distance,
        "elevation_deg": pose.elevation_deg,
        "lateral_offset": pose.lateral_offset,
        "z_offset": pose.z_offset,
        "up_variant": pose.up_variant,
    }
    payload = camera_visibility_cache_key_payload(**kwargs)
    return camera_visibility_cache_key(**kwargs), payload


def _workbook_provenance(job: SequenceJob) -> dict[str, Any]:
    return {
        "workbook_name": Path(job.workbook_path).name,
        "workbook_sha256": job.workbook_sha256,
        "sequence_sheet": "sequences",
        "sequence_excel_row": job.source_excel_row,
    }


def _disabled_cache_event(kind: str, cache_id: str, reason: str) -> CacheEvent:
    return CacheEvent(
        kind=kind,
        cache_id=cache_id,
        status="disabled",
        reason=reason,
        payload_path=None,
        sidecar_path=None,
    )


def _supports_persistent_cache(backend: Any) -> bool:
    required_methods = (
        "prepare_clean_context",
        "compute_clean",
        "serialize_clean",
        "restore_clean",
        "serialize_particle",
        "restore_particle",
    )
    return all(callable(getattr(backend, name, None)) for name in required_methods)


def _mark_invalid_hit(event: CacheEvent) -> CacheEvent:
    return CacheEvent(
        kind=event.kind,
        cache_id=event.cache_id,
        status="miss",
        reason="miss_payload_invalid",
        payload_path=event.payload_path,
        sidecar_path=event.sidecar_path,
        load_seconds=event.load_seconds,
    )


def _summarize_pose_cache_events(kind: str, events: list[CacheEvent]) -> CacheEvent:
    if not events:
        return _disabled_cache_event(kind, "", "no_poses")
    n_hits = sum(1 for event in events if event.status == "hit")
    n_misses = len(events) - n_hits
    load_seconds = float(sum(event.load_seconds for event in events))
    write_seconds = float(sum(event.write_seconds for event in events))
    if n_misses == 0:
        status = "hit"
        reason = f"all_hit;n_poses={len(events)}"
    elif n_hits == 0:
        status = "miss"
        reason = f"all_miss;n_poses={len(events)};first={events[0].reason}"
    else:
        status = "mixed"
        reason = f"n_hits={n_hits};n_misses={n_misses}"
    return CacheEvent(
        kind=kind,
        cache_id="aggregate",
        status=status,
        reason=reason,
        payload_path=None,
        sidecar_path=None,
        load_seconds=load_seconds,
        write_seconds=write_seconds,
    )


def _summarize_visibility_events(events: list[CacheEvent]) -> CacheEvent:
    return _summarize_pose_cache_events("camera_visibility", events)


def _load_or_compute_camera_visibility(
    *,
    surface_mesh: Any,
    rays: Any,
    job: SequenceJob,
    pose: CameraPose,
    settings: SmokeRuntimeSettings,
    cache_store: SourceCacheStore | None,
    force_recompute: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CacheEvent]:
    """Return first-surface hits, optionally via pose×mesh visibility cache."""
    cache_id, key_payload = _camera_visibility_key_parts(job, pose, settings)
    if cache_store is None:
        valid, depth, faces, points = first_visible_hits_with_points(
            surface_mesh,
            rays,
        )
        return (
            valid,
            depth,
            faces,
            points,
            _disabled_cache_event(
                "camera_visibility",
                cache_id,
                "cache_disabled",
            ),
        )

    surface_identity = _camera_visibility_surface_identity(job)
    loaded = cache_store.load(
        kind="camera_visibility",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=CAMERA_VISIBILITY_PAYLOAD_SCHEMA_VERSION,
        required_arrays=CAMERA_VISIBILITY_REQUIRED_ARRAYS,
        mesh_identity=surface_identity,
        force_recompute=force_recompute,
        allow_nonfinite_arrays=CAMERA_VISIBILITY_ALLOW_NONFINITE_ARRAYS,
    )
    if loaded.event.hit:
        assert loaded.arrays is not None
        arrays = loaded.arrays
        expected_shape = tuple(int(x) for x in arrays["sample_shape"].tolist())
        if not expected_shape or expected_shape == tuple(rays.sample_shape):
            return (
                np.asarray(arrays["valid_mask"]),
                np.asarray(arrays["hit_depth"]),
                np.asarray(arrays["hit_faces"]),
                np.asarray(arrays["hit_points"]),
                loaded.event,
            )

    valid, depth, faces, points = first_visible_hits_with_points(
        surface_mesh,
        rays,
    )
    event = loaded.event
    if loaded.event.hit:
        event = _mark_invalid_hit(loaded.event)
    arrays = {
        "valid_mask": np.asarray(valid, dtype=bool),
        "hit_depth": np.asarray(depth, dtype=float),
        "hit_faces": np.asarray(faces, dtype=np.int64),
        "hit_points": np.asarray(points, dtype=float),
        "sample_shape": _sample_shape_array(rays.sample_shape),
    }
    event = cache_store.write(
        kind="camera_visibility",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=CAMERA_VISIBILITY_PAYLOAD_SCHEMA_VERSION,
        arrays=arrays,
        payload_metadata={
            "look_at_policy": CAMERA_LOOK_AT_POLICY,
            "frame_index": int(pose.frame_index),
            "angle_deg": float(pose.angle_deg),
        },
        mesh_identity=surface_identity,
        workbook_provenance=_workbook_provenance(job),
        prior_event=event,
    )
    return valid, depth, faces, points, event


def _phi_sampling_localization_key_parts(
    *,
    camera_visibility_cache_id: str,
    diff_mesh: Any,
) -> tuple[str, dict[str, Any]]:
    kwargs = {
        "camera_visibility_cache_id": camera_visibility_cache_id,
        "diffusion_mesh_content_hash": str(diff_mesh.content_hash()),
        "diffusion_mesh_num_nodes": int(diff_mesh.n_nodes),
        "diffusion_mesh_num_tets": int(diff_mesh.n_tets),
        "barycentric_tolerance": DEFAULT_BARYCENTRIC_TOLERANCE,
    }
    payload = phi_sampling_localization_cache_key_payload(**kwargs)
    return phi_sampling_localization_cache_key(**kwargs), payload


def _load_or_compute_phi_localization(
    *,
    diff_mesh: Any,
    hit_points: np.ndarray,
    valid_mask: np.ndarray,
    sample_shape: tuple[int, ...],
    camera_visibility_cache_id: str,
    job: SequenceJob,
    cache_store: SourceCacheStore | None,
    force_recompute: bool,
) -> tuple[PhiSamplingLocalization, CacheEvent]:
    """Return tet/barycentric fluence field ``Phi`` localization, optionally from disk cache."""
    cache_id, key_payload = _phi_sampling_localization_key_parts(
        camera_visibility_cache_id=camera_visibility_cache_id,
        diff_mesh=diff_mesh,
    )
    mesh_identity = _mesh_identity(diff_mesh)
    if cache_store is None:
        localization = localize_points_in_diffusion_mesh(
            diff_mesh,
            hit_points,
            valid_mask=valid_mask,
            barycentric_tolerance=DEFAULT_BARYCENTRIC_TOLERANCE,
        )
        return (
            localization,
            _disabled_cache_event(
                "phi_sampling_localization",
                cache_id,
                "cache_disabled",
            ),
        )

    loaded = cache_store.load(
        kind="phi_sampling_localization",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=PHI_SAMPLING_LOCALIZATION_PAYLOAD_SCHEMA_VERSION,
        required_arrays=PHI_SAMPLING_LOCALIZATION_REQUIRED_ARRAYS,
        mesh_identity=mesh_identity,
        force_recompute=force_recompute,
    )
    if loaded.event.hit:
        assert loaded.arrays is not None
        arrays = loaded.arrays
        expected_shape = tuple(int(x) for x in arrays["sample_shape"].tolist())
        if not expected_shape or expected_shape == tuple(sample_shape):
            localization = PhiSamplingLocalization(
                sample_mode=np.asarray(arrays["sample_mode"], dtype=np.int8),
                tet_id=np.asarray(arrays["tet_id"], dtype=np.int64),
                barycentric=np.asarray(arrays["barycentric"], dtype=float),
            )
            return localization, loaded.event

    localization = localize_points_in_diffusion_mesh(
        diff_mesh,
        hit_points,
        valid_mask=valid_mask,
        barycentric_tolerance=DEFAULT_BARYCENTRIC_TOLERANCE,
    )
    event = loaded.event
    if loaded.event.hit:
        event = _mark_invalid_hit(loaded.event)
    arrays = {
        "sample_mode": np.asarray(localization.sample_mode, dtype=np.int8),
        "tet_id": np.asarray(localization.tet_id, dtype=np.int64),
        "barycentric": np.asarray(localization.barycentric, dtype=float),
        "sample_shape": _sample_shape_array(sample_shape),
    }
    event = cache_store.write(
        kind="phi_sampling_localization",
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=PHI_SAMPLING_LOCALIZATION_PAYLOAD_SCHEMA_VERSION,
        arrays=arrays,
        payload_metadata={
            "barycentric_tolerance": DEFAULT_BARYCENTRIC_TOLERANCE,
            "interpolation_method": "tetrahedral_barycentric",
        },
        mesh_identity=mesh_identity,
        workbook_provenance=_workbook_provenance(job),
        prior_event=event,
    )
    return localization, event


def _particle_ray_weights(clean: _CleanState, particle: _ParticleState) -> np.ndarray:
    weights = np.asarray(clean.source_rays.weights, dtype=float).copy()
    for pair in particle.pair_result.pairs:
        path_id = int(pair.path_id)
        if path_id < 0 or path_id >= len(weights):
            raise ValueError(
                f"Affected path_id={path_id} does not index source ray weights."
            )
        attenuation = 1.0
        for event in pair.particle_events:
            inclusion = particle.particles[event.particle_index]
            attenuation *= np.exp(
                -inclusion.mu_total * float(event.path_length_inside_particle)
            )
        weights[path_id] *= attenuation
    return weights


def _orbit_camera(
    pose: CameraPose,
    *,
    look_at: np.ndarray,
    fov_deg: float,
) -> PinholeCameraConfig:
    if pose.resolution_x != pose.resolution_y:
        raise ValueError(
            "Phase 2 pinhole capture requires square resolution; got "
            f"{pose.resolution_x}x{pose.resolution_y}."
        )
    axis = np.asarray(pose.axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("Camera orbit axis must be nonzero.")
    axis /= axis_norm

    reference = np.array([0.0, -1.0, 0.0])
    reference -= float(np.dot(reference, axis)) * axis
    if np.linalg.norm(reference) < 1e-8:
        reference = np.array([1.0, 0.0, 0.0])
        reference -= float(np.dot(reference, axis)) * axis
    reference /= np.linalg.norm(reference)
    tangent = np.cross(axis, reference)

    angle = np.deg2rad(float(pose.angle_deg))
    elevation = np.deg2rad(float(pose.elevation_deg))
    radial = np.cos(angle) * reference + np.sin(angle) * tangent
    offset = float(pose.distance) * (
        np.cos(elevation) * radial + np.sin(elevation) * axis
    )
    camera_position = look_at + offset
    up = axis
    view = look_at - camera_position
    if abs(float(np.dot(view / np.linalg.norm(view), up))) > 0.98:
        up = tangent

    return PinholeCameraConfig(
        camera_position=tuple(float(x) for x in camera_position),
        look_at=tuple(float(x) for x in look_at),
        up=tuple(float(x) for x in up),
        fov_deg=float(fov_deg),
        resolution=int(pose.resolution_x),
    )


class DefaultSmokePhysicsBackend:
    """Production physics backend for multi-view sequence generation.

    Wires refractive source transport, volumetric finite-element method (FEM) diffusion, and particle
    source correction into the :class:`SmokePhysicsBackend` lifecycle.
    Supports persistent clean and particle source cache serialize/restore.
    """

    def prepare_clean_context(
        self,
        job: SequenceJob,
        settings: SmokeRuntimeSettings,
    ) -> _CleanContext:
        """Load surface STL triangle mesh file, build diffusion mesh, and resolve optical configs.

        Mesh and material objects are shared across cache restore paths.
        Requires point illumination and exact ray-tet source deposition.
        """
        if job.optical.illumination_kind != "point":
            raise ValueError("M6 Phase 2 supports illumination_kind='point' only.")
        if job.optical.source_deposition_method != "exact_ray_tet_intervals":
            raise ValueError(
                "M6 Phase 2 requires source_deposition_method="
                "'exact_ray_tet_intervals'."
            )
        stl_path = _resolve_job_path(job)
        surface_mesh = load_stl(stl_path)
        diff_mesh = generate_diffusion_mesh(
            stl_path,
            target_elements=int(settings.target_elements),
        )
        material = job.optical.as_optical_material(g=job.diffusion.g)
        light = PointLightConfig(
            position=(
                job.optical.light_position_x,
                job.optical.light_position_y,
                job.optical.light_position_z,
            ),
            intensity=float(job.optical.source_intensity),
        )
        return _CleanContext(
            surface_mesh=surface_mesh,
            diff_mesh=diff_mesh,
            material=material,
            light=light,
        )

    def compute_clean(
        self,
        job: SequenceJob,
        context: _CleanContext,
        settings: SmokeRuntimeSettings,
    ) -> _CleanState:
        """Trace, refract, and deposit the clean optical source on ``context``.

        Returns:
            Full clean state including ray bundles, segments, and ``S_clean``.
        """
        source_rays = make_source_ray_bundle(
            context.light,
            context.surface_mesh.bounds,
            SourceSamplingParams(
                mode="point_uniform",
                n_rays=job.optical.num_source_rays,
                # Unspecified RNG: do not couple illumination sampling to
                # sequences.seed (reserved for other workbook uses).
                seed=None,
            ),
        )
        refracted = refract_ray_bundle(
            context.surface_mesh,
            source_rays,
            n_from=settings.n_from,
            n_to=context.material.n_refractive,
        )
        segments = in_object_segments_from_rays(
            context.surface_mesh,
            refracted.rays,
            parent_ray_ids=refracted.parent_indices,
        )
        if segments.n_segments <= 0:
            raise RuntimeError("No in-object source segments were generated.")
        deposition = deposit_ray_source(
            context.diff_mesh,
            segments,
            material=context.material,
        )
        return _CleanState(
            surface_mesh=context.surface_mesh,
            diff_mesh=context.diff_mesh,
            material=context.material,
            light=context.light,
            source_rays=source_rays,
            refracted=refracted,
            segments=segments,
            deposition=deposition,
        )

    def prepare_clean(
        self,
        job: SequenceJob,
        settings: SmokeRuntimeSettings,
    ) -> _CleanState:
        """Build clean source state (context + deposition) when caching is off."""
        context = self.prepare_clean_context(job, settings)
        return self.compute_clean(job, context, settings)

    def serialize_clean(
        self,
        state: _CleanState,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Pack clean state into cacheable numpy arrays and sidecar metadata."""
        arrays = {
            "source_origins": state.source_rays.origins,
            "source_directions": state.source_rays.directions,
            "source_weights": state.source_rays.weights,
            "source_sample_shape": _sample_shape_array(state.source_rays.sample_shape),
            "refracted_origins": state.refracted.rays.origins,
            "refracted_directions": state.refracted.rays.directions,
            "refracted_weights": state.refracted.rays.weights,
            "refracted_sample_shape": _sample_shape_array(
                state.refracted.rays.sample_shape
            ),
            "refracted_parent_indices": state.refracted.parent_indices,
            "refracted_valid_mask": state.refracted.valid_mask,
            "refracted_hit_faces": state.refracted.hit_faces,
            "refracted_hit_points": state.refracted.hit_points,
            "refracted_full_directions": state.refracted.refracted_directions,
            "segment_starts": state.segments.starts,
            "segment_ends": state.segments.ends,
            "segment_intensities": state.segments.intensities,
            "segment_ray_ids": state.segments.ray_ids,
            "segment_ids": state.segments.segment_ids,
            "segment_path_order": state.segments.path_order,
            "S_clean": state.deposition.S_clean,
            "E_scat_elem": state.deposition.E_scat_elem,
        }
        metadata = {
            "refracted_eps": float(state.refracted.eps),
            "deposition": {
                "total_ballistic_input": float(state.deposition.total_ballistic_input),
                "total_scattered": float(state.deposition.total_scattered),
                "total_absorbed": float(state.deposition.total_absorbed),
                "remaining_direct_energy": float(
                    state.deposition.remaining_direct_energy
                ),
                "mu_s": float(state.deposition.mu_s),
                "mu_a": float(state.deposition.mu_a),
            },
        }
        return arrays, metadata

    def restore_clean(
        self,
        context: _CleanContext,
        arrays: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> _CleanState:
        """Reconstruct clean state from cached arrays plus shared ``context``."""
        source_rays = SourceRayBundle(
            origins=arrays["source_origins"],
            directions=arrays["source_directions"],
            weights=arrays["source_weights"],
            sample_shape=_sample_shape_from_array(arrays["source_sample_shape"]),
        )
        refracted_rays = SourceRayBundle(
            origins=arrays["refracted_origins"],
            directions=arrays["refracted_directions"],
            weights=arrays["refracted_weights"],
            sample_shape=_sample_shape_from_array(arrays["refracted_sample_shape"]),
        )
        refracted = RefractedRayBundleResult(
            rays=refracted_rays,
            parent_indices=arrays["refracted_parent_indices"],
            valid_mask=arrays["refracted_valid_mask"],
            hit_faces=arrays["refracted_hit_faces"],
            hit_points=arrays["refracted_hit_points"],
            refracted_directions=arrays["refracted_full_directions"],
            eps=float(metadata["refracted_eps"]),
        )
        segments = RaySegmentBundle(
            starts=arrays["segment_starts"],
            ends=arrays["segment_ends"],
            intensities=arrays["segment_intensities"],
            ray_ids=arrays["segment_ray_ids"],
            segment_ids=arrays["segment_ids"],
            path_order=arrays["segment_path_order"],
        )
        deposition_metadata = metadata["deposition"]
        deposition = SourceDepositionResult(
            S_clean=arrays["S_clean"],
            E_scat_elem=arrays["E_scat_elem"],
            total_ballistic_input=float(deposition_metadata["total_ballistic_input"]),
            total_scattered=float(deposition_metadata["total_scattered"]),
            total_absorbed=float(deposition_metadata["total_absorbed"]),
            remaining_direct_energy=float(
                deposition_metadata["remaining_direct_energy"]
            ),
            mu_s=float(deposition_metadata["mu_s"]),
            mu_a=float(deposition_metadata["mu_a"]),
        )
        return _CleanState(
            surface_mesh=context.surface_mesh,
            diff_mesh=context.diff_mesh,
            material=context.material,
            light=context.light,
            source_rays=source_rays,
            refracted=refracted,
            segments=segments,
            deposition=deposition,
        )

    def _particles_for_job(self, job: SequenceJob) -> ParticleSet:
        return ParticleSet.from_particles(
            [
                ParticleSphere(
                    center=(
                        item.center_x,
                        item.center_y,
                        item.center_z,
                    ),
                    radius=item.radius,
                    mu_abs=item.mu_a_particle,
                    mu_scat=item.mu_s_particle,
                    particle_id=item.particle_setup_id,
                )
                for item in job.particles
            ],
            metadata={
                "placement_mode": job.particle.placement_mode,
                "particle_group_id": job.particle_group_id,
                "particle_count": len(job.particles),
            },
        )

    def prepare_particle(
        self,
        job: SequenceJob,
        clean_state: _CleanState,
        settings: SmokeRuntimeSettings,
    ) -> _ParticleState:
        """Build affected transport pairs and particle diffusion source correction.

        Warns when configured particles miss all source transport paths.
        """
        for item in job.particles:
            if item.particle_kind != "sphere":
                raise ValueError(
                    "M6 supports particle_kind='sphere' only "
                    f"(got {item.particle_kind!r})."
                )
            if item.placement_mode != "fixed":
                raise ValueError(
                    "M6 supports placement_mode='fixed' only "
                    f"(got {item.placement_mode!r})."
                )
        particles = self._particles_for_job(job)
        pairs = build_affected_transport_pairs(
            clean_state.segments,
            particles,
            material=clean_state.material,
        )
        notes: list[str] = []
        no_transport_intersection = not pairs.pairs
        if no_transport_intersection:
            centers = "; ".join(
                "[" + ", ".join(f"{float(v):.6g}" for v in particle.center) + "]"
                for particle in particles
            )
            message = (
                "Configured particle intersects no source transport paths "
                f"(sequence_id={job.sequence_id!r}, centers={centers}). "
                "Treating particle source as identical to clean "
                "(shadow / refractive miss). Inspect placement if unexpected."
            )
            warnings.warn(message, UserWarning, stacklevel=2)
            notes.append(message)
        correction = compute_transport_source_correction(
            clean_state.diff_mesh,
            pairs,
            clean_state.deposition.E_scat_elem,
            material=clean_state.material,
            assignment=settings.source_delta_assignment,
        )
        state = _ParticleState(
            particles=particles,
            pair_result=pairs,
            source_correction=correction,
            no_transport_intersection=no_transport_intersection,
            notes=tuple(notes),
        )
        state.particle_ray_weights = _particle_ray_weights(clean_state, state)
        return state

    def serialize_particle(
        self,
        state: _ParticleState,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Pack particle source correction and ray weights for disk cache."""
        correction = state.source_correction
        arrays = {
            "E_clean_elem": correction.E_clean_elem,
            "delta_E_background_elem": correction.delta_E_background_elem,
            "delta_E_particle_scat_elem": (correction.delta_E_particle_scat_elem),
            "delta_E_transport_elem": correction.delta_E_transport_elem,
            "E_particle_elem": correction.E_particle_elem,
            "S_particle": correction.S_particle,
            "affected_path_ids": np.asarray(
                state.pair_result.affected_path_ids,
                dtype=np.int64,
            ),
            "affected_segment_indices": (state.pair_result.affected_segment_indices),
            "particle_ray_weights": state.particle_ray_weights,
        }
        metadata = {
            "source_assignment": (
                correction.particle_scatter_deposition.assignment_mode
            ),
            "n_affected_paths": len(state.pair_result.affected_path_ids),
            "no_transport_intersection": bool(state.no_transport_intersection),
            "notes": list(state.notes),
        }
        return arrays, metadata

    def restore_particle(
        self,
        job: SequenceJob,
        arrays: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> _ParticleState:
        """Reconstruct particle state from cache; rebinds particles from ``job``."""
        pair_summary = _CachedPairSummary(
            affected_path_ids=tuple(
                int(value) for value in arrays["affected_path_ids"].tolist()
            ),
            affected_segment_indices=arrays["affected_segment_indices"],
        )
        correction = _CachedSourceCorrection(
            E_clean_elem=arrays["E_clean_elem"],
            delta_E_background_elem=arrays["delta_E_background_elem"],
            delta_E_particle_scat_elem=arrays["delta_E_particle_scat_elem"],
            delta_E_transport_elem=arrays["delta_E_transport_elem"],
            E_particle_elem=arrays["E_particle_elem"],
            S_particle=arrays["S_particle"],
            particle_scatter_deposition=_CachedParticleScatterSummary(
                assignment_mode=str(metadata["source_assignment"])
            ),
        )
        n_affected = len(pair_summary.affected_path_ids)
        no_transport = bool(
            metadata.get("no_transport_intersection", n_affected == 0)
        )
        notes = tuple(str(item) for item in metadata.get("notes", ()) or ())
        if no_transport and not notes:
            notes = (
                "Cached particle source has no affected transport paths "
                f"(sequence_id={job.sequence_id!r}).",
            )
        return _ParticleState(
            particles=self._particles_for_job(job),
            pair_result=pair_summary,
            source_correction=correction,
            particle_ray_weights=arrays["particle_ray_weights"],
            no_transport_intersection=no_transport,
            notes=notes,
        )

    def solve_fields(
        self,
        job: SequenceJob,
        clean_state: _CleanState,
        particle_state: _ParticleState,
        settings: SmokeRuntimeSettings,
    ) -> _FieldState:
        """Solve volumetric diffusion (fluence field ``Phi``) for clean and particle source densities ``S(x)``."""
        solve_kwargs = {
            "D": float(job.diffusion_provenance["D"]),
            "mu_a": float(job.diffusion_provenance["mu_a"]),
            "extrapolation_length": job.diffusion.extrapolation_length,
            "robin_boundary_model": job.diffusion.robin_boundary_model,
            "fem_order": job.diffusion.fem_order,
        }
        clean_solve = solve_diffusion(
            clean_state.diff_mesh,
            clean_state.deposition.S_clean,
            **solve_kwargs,
        )
        particle_solve = solve_diffusion(
            clean_state.diff_mesh,
            particle_state.source_correction.S_particle,
            **solve_kwargs,
        )
        return _FieldState(
            clean_solve=clean_solve,
            particle_solve=particle_solve,
        )

    def capture_frame(
        self,
        job: SequenceJob,
        pose: CameraPose,
        clean_state: _CleanState,
        particle_state: _ParticleState,
        field_state: _FieldState,
        settings: SmokeRuntimeSettings,
        *,
        visibility_cache_store: SourceCacheStore | None = None,
        force_recompute_visibility: bool = False,
    ) -> CapturedFrame:
        """Sample diffuse fluence field ``Phi`` (and optional direct) camera intensity for one pose.

        Uses optional visibility and ``Phi``-localization caches when provided.
        """
        look_at = np.asarray(clean_state.surface_mesh.bounds, dtype=float).mean(axis=0)
        camera = _orbit_camera(
            pose,
            look_at=look_at,
            fov_deg=settings.camera_fov_deg,
        )
        rays = make_camera_rays(camera)
        valid, _depth, faces, points, visibility_event = (
            _load_or_compute_camera_visibility(
                surface_mesh=clean_state.surface_mesh,
                rays=rays,
                job=job,
                pose=pose,
                settings=settings,
                cache_store=visibility_cache_store,
                force_recompute=force_recompute_visibility,
            )
        )
        sample_shape = rays.sample_shape
        localization, localization_event = _load_or_compute_phi_localization(
            diff_mesh=clean_state.diff_mesh,
            hit_points=points,
            valid_mask=valid,
            sample_shape=sample_shape,
            camera_visibility_cache_id=visibility_event.cache_id,
            job=job,
            cache_store=visibility_cache_store,
            force_recompute=force_recompute_visibility,
        )
        clean_diffuse = sample_diffuse_image(
            clean_state.diff_mesh,
            field_state.clean_solve.Phi_nodes,
            points,
            valid,
            sample_shape,
            exitance_scale=settings.exitance_scale,
            interpolate=True,
            localization=localization,
        ).I_diffuse
        particle_diffuse = sample_diffuse_image(
            clean_state.diff_mesh,
            field_state.particle_solve.Phi_nodes,
            points,
            valid,
            sample_shape,
            exitance_scale=settings.exitance_scale,
            interpolate=True,
            localization=localization,
        ).I_diffuse

        clean_image = clean_diffuse
        particle_image = particle_diffuse
        if job.diffusion.alpha_direct != 0.0:
            height, width = sample_shape
            mask = valid.reshape(height, width)
            hit_faces = faces.reshape(height, width)
            view_directions = -rays.directions.reshape(height, width, 3)
            clean_direct = compute_refractive_direct_image(
                clean_state.surface_mesh,
                clean_state.source_rays,
                clean_state.material,
                hit_faces,
                view_directions,
                apply_attenuation=True,
                camera_mask=mask,
            ).I_direct
            particle_direct = compute_refractive_direct_image(
                clean_state.surface_mesh,
                clean_state.source_rays,
                clean_state.material,
                hit_faces,
                view_directions,
                ray_weights=particle_state.particle_ray_weights,
                apply_attenuation=True,
                camera_mask=mask,
            ).I_direct
            clean_image = compose_hybrid_image(
                clean_direct,
                clean_diffuse,
                alpha=job.diffusion.alpha_direct,
                camera_mask=mask,
            ).I_total
            particle_image = compose_hybrid_image(
                particle_direct,
                particle_diffuse,
                alpha=job.diffusion.alpha_direct,
                camera_mask=mask,
            ).I_total

        return CapturedFrame(
            clean=np.asarray(clean_image, dtype=float),
            particle=np.asarray(particle_image, dtype=float),
            metadata={
                "frame_index": pose.frame_index,
                "angle_deg": pose.angle_deg,
                "axis": list(pose.axis),
                "camera_position": list(camera.camera_position),
                "look_at": list(camera.look_at),
                "up": list(camera.up),
                "camera_kind": pose.camera_kind,
                "resolution": list(sample_shape),
                "fov_deg": camera.fov_deg,
                "visibility_cache": {
                    "cache_id": visibility_event.cache_id,
                    "status": visibility_event.status,
                    "reason": visibility_event.reason,
                },
                "phi_sampling_localization_cache": {
                    "cache_id": localization_event.cache_id,
                    "status": localization_event.status,
                    "reason": localization_event.reason,
                },
            },
        )

    def diagnostics(
        self,
        clean_state: _CleanState,
        particle_state: _ParticleState,
        field_state: _FieldState,
    ) -> dict[str, Any]:
        """Return ray, transport, and diffusion solve counts for the manifest."""
        return {
            "n_source_rays": int(clean_state.source_rays.n_rays),
            "n_refracted_rays": int(clean_state.refracted.n_refracted),
            "n_source_segments": int(clean_state.segments.n_segments),
            "n_affected_paths": int(len(particle_state.pair_result.affected_path_ids)),
            "no_transport_intersection": bool(
                particle_state.no_transport_intersection
            ),
            "notes": list(particle_state.notes),
            "source_assignment": (
                particle_state.source_correction.particle_scatter_deposition.assignment_mode
            ),
            "clean_solve_residual": float(field_state.clean_solve.residual_norm),
            "particle_solve_residual": float(field_state.particle_solve.residual_norm),
        }


def generate_smoke_sequence(
    job: SequenceJob,
    *,
    output_root: Path | str | None = None,
    cache_root: Path | str | None = None,
    force_recompute: bool = False,
    backend: SmokePhysicsBackend | None = None,
    settings: SmokeRuntimeSettings | None = None,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    write_anomaly_preview: bool = True,
    max_workers: int = 1,
) -> GeneratedSequenceResult:
    """Generate one multi-view sequence with persistent source caches.

    Pipeline: resolve output identity → load/compute clean optical cache →
    load/compute particle source cache → solve clean and particle diffusion →
    capture each camera pose → write role directories and ``manifest.json``.

    Args:
        job: Planned sequence job with cache ids and camera poses.
        output_root: Optional override of job ``output_root``.
        cache_root: Persistent ``_cache`` directory (``None`` disables reuse).
        force_recompute: Rebuild caches even when valid pairs exist.
        backend: Physics backend (default: :class:`DefaultSmokePhysicsBackend`).
        settings: Runtime knobs recorded in the manifest.
        image_format, jpeg_quality, write_anomaly_preview: Role encoding options.
        max_workers: Reserved for intra-sequence parallelism (currently serial).

    Returns:
        GeneratedSequenceResult with paths, cache events, and timings.

    Notebook / protocol:
        Lower-level entry when :func:`run_generation_plan` batching is not needed.
    """
    backend = DefaultSmokePhysicsBackend() if backend is None else backend
    settings = SmokeRuntimeSettings() if settings is None else settings
    job = resolve_job_output_identity(
        job,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        write_anomaly_preview=write_anomaly_preview,
        runtime_settings=asdict(settings),
    )
    if settings.target_elements != 1000:
        raise ValueError("M6 Phase 2 clean cache identity fixes target_elements=1000.")
    if settings.n_from != 1.0:
        raise ValueError("M6 Phase 2 clean cache identity fixes n_from=1.0.")
    if settings.source_delta_assignment != "attenuated_chord":
        raise ValueError(
            "M6 Phase 2 particle cache identity requires "
            "source_delta_assignment='attenuated_chord'."
        )
    if settings.surface_interpolation != "tetrahedral_barycentric":
        raise ValueError("M6 Phase 2 requires tetrahedral barycentric camera sampling.")
    stage_seconds: dict[str, float] = {}

    started = perf_counter()
    cache_enabled = cache_root is not None and _supports_persistent_cache(backend)
    cache_store: SourceCacheStore | None = None
    if cache_enabled:
        cache_backend: Any = backend
        assert cache_root is not None
        cache_store = SourceCacheStore(cache_root)
        context = cache_backend.prepare_clean_context(job, settings)
        mesh_identity = _mesh_identity(context.diff_mesh)
        clean_load = cache_store.load(
            kind="clean_optical",
            cache_id=job.clean_optical_cache_id,
            key_payload=job.clean_optical_cache_payload,
            payload_schema_version=CLEAN_PAYLOAD_SCHEMA_VERSION,
            required_arrays=CLEAN_REQUIRED_ARRAYS,
            mesh_identity=mesh_identity,
            force_recompute=force_recompute,
            allow_nonfinite_arrays=CLEAN_ALLOW_NONFINITE_ARRAYS,
        )
        clean_cache = clean_load.event
        if clean_load.event.hit:
            try:
                assert clean_load.arrays is not None
                assert clean_load.payload_metadata is not None
                clean_state = cache_backend.restore_clean(
                    context,
                    clean_load.arrays,
                    clean_load.payload_metadata,
                )
            except (KeyError, TypeError, ValueError):
                clean_cache = _mark_invalid_hit(clean_load.event)
                clean_state = cache_backend.compute_clean(job, context, settings)
        else:
            clean_state = cache_backend.compute_clean(job, context, settings)
        if not clean_load.event.hit or clean_cache.status == "miss":
            clean_arrays, clean_metadata = cache_backend.serialize_clean(clean_state)
            clean_cache = cache_store.write(
                kind="clean_optical",
                cache_id=job.clean_optical_cache_id,
                key_payload=job.clean_optical_cache_payload,
                payload_schema_version=CLEAN_PAYLOAD_SCHEMA_VERSION,
                arrays=clean_arrays,
                payload_metadata=clean_metadata,
                mesh_identity=mesh_identity,
                workbook_provenance=_workbook_provenance(job),
                prior_event=clean_cache,
            )
    else:
        clean_state = backend.prepare_clean(job, settings)
        mesh_identity = None
        clean_cache = _disabled_cache_event(
            "clean_optical",
            job.clean_optical_cache_id,
            (
                "cache_root_not_configured"
                if cache_root is None
                else "backend_not_cacheable"
            ),
        )
    stage_seconds["clean_source"] = perf_counter() - started

    started = perf_counter()
    if cache_enabled:
        assert mesh_identity is not None
        particle_load = cache_store.load(
            kind="particle_source",
            cache_id=job.particle_source_cache_id,
            key_payload=job.particle_source_cache_payload,
            payload_schema_version=PARTICLE_PAYLOAD_SCHEMA_VERSION,
            required_arrays=PARTICLE_REQUIRED_ARRAYS,
            mesh_identity=mesh_identity,
            force_recompute=force_recompute,
        )
        particle_cache = particle_load.event
        if particle_load.event.hit:
            try:
                assert particle_load.arrays is not None
                assert particle_load.payload_metadata is not None
                particle_state = cache_backend.restore_particle(
                    job,
                    particle_load.arrays,
                    particle_load.payload_metadata,
                )
            except (KeyError, TypeError, ValueError):
                particle_cache = _mark_invalid_hit(particle_load.event)
                particle_state = backend.prepare_particle(
                    job,
                    clean_state,
                    settings,
                )
        else:
            particle_state = backend.prepare_particle(job, clean_state, settings)
        if not particle_load.event.hit or particle_cache.status == "miss":
            particle_arrays, particle_metadata = cache_backend.serialize_particle(
                particle_state
            )
            particle_cache = cache_store.write(
                kind="particle_source",
                cache_id=job.particle_source_cache_id,
                key_payload=job.particle_source_cache_payload,
                payload_schema_version=PARTICLE_PAYLOAD_SCHEMA_VERSION,
                arrays=particle_arrays,
                payload_metadata=particle_metadata,
                mesh_identity=mesh_identity,
                workbook_provenance=_workbook_provenance(job),
                prior_event=particle_cache,
            )
    else:
        particle_state = backend.prepare_particle(job, clean_state, settings)
        particle_cache = _disabled_cache_event(
            "particle_source",
            job.particle_source_cache_id,
            clean_cache.reason,
        )
    stage_seconds["particle_source"] = perf_counter() - started

    started = perf_counter()
    field_state = backend.solve_fields(
        job,
        clean_state,
        particle_state,
        settings,
    )
    stage_seconds["diffusion_solves"] = perf_counter() - started

    captures: list[CapturedFrame] = []
    visibility_events: list[CacheEvent] = []
    localization_events: list[CacheEvent] = []
    started = perf_counter()
    visibility_store = cache_store if cache_enabled else None
    capture_params = signature(backend.capture_frame).parameters
    supports_visibility_kwargs = "visibility_cache_store" in capture_params or any(
        parameter.kind == parameter.VAR_KEYWORD
        for parameter in capture_params.values()
    )
    for pose in job.camera.poses:
        capture_kwargs: dict[str, Any] = {}
        if visibility_store is not None and supports_visibility_kwargs:
            capture_kwargs = {
                "visibility_cache_store": visibility_store,
                "force_recompute_visibility": force_recompute,
            }
        capture = backend.capture_frame(
            job,
            pose,
            clean_state,
            particle_state,
            field_state,
            settings,
            **capture_kwargs,
        )
        captures.append(capture)
        frame_vis = capture.metadata.get("visibility_cache")
        if isinstance(frame_vis, dict):
            visibility_events.append(
                CacheEvent(
                    kind="camera_visibility",
                    cache_id=str(frame_vis.get("cache_id", "")),
                    status=str(frame_vis.get("status", "miss")),
                    reason=str(frame_vis.get("reason", "")),
                    payload_path=None,
                    sidecar_path=None,
                )
            )
        frame_loc = capture.metadata.get("phi_sampling_localization_cache")
        if isinstance(frame_loc, dict):
            localization_events.append(
                CacheEvent(
                    kind="phi_sampling_localization",
                    cache_id=str(frame_loc.get("cache_id", "")),
                    status=str(frame_loc.get("status", "miss")),
                    reason=str(frame_loc.get("reason", "")),
                    payload_path=None,
                    sidecar_path=None,
                )
            )
    stage_seconds["camera_capture"] = perf_counter() - started
    if visibility_store is None:
        visibility_cache = _disabled_cache_event(
            "camera_visibility",
            "",
            "cache_disabled" if cache_root is None else "backend_without_source_cache",
        )
        phi_localization_cache = _disabled_cache_event(
            "phi_sampling_localization",
            "",
            "cache_disabled" if cache_root is None else "backend_without_source_cache",
        )
    else:
        visibility_cache = (
            _summarize_pose_cache_events("camera_visibility", visibility_events)
            if visibility_events
            else _disabled_cache_event(
                "camera_visibility",
                "",
                "backend_without_visibility_cache",
            )
        )
        phi_localization_cache = (
            _summarize_pose_cache_events(
                "phi_sampling_localization",
                localization_events,
            )
            if localization_events
            else _disabled_cache_event(
                "phi_sampling_localization",
                "",
                "backend_without_phi_localization_cache",
            )
        )

    frame_metadata = [dict(capture.metadata) for capture in captures]
    manifest = build_sequence_manifest(
        job,
        frame_metadata=frame_metadata,
        runtime_settings={
            "target_elements": settings.target_elements,
            "exitance_scale": settings.exitance_scale,
            "camera_fov_deg": settings.camera_fov_deg,
            "n_from": settings.n_from,
            "source_delta_assignment": settings.source_delta_assignment,
            "surface_interpolation": settings.surface_interpolation,
        },
        stage_seconds=stage_seconds,
        diagnostics=backend.diagnostics(
            clean_state,
            particle_state,
            field_state,
        ),
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        anomaly_preview=write_anomaly_preview,
        cache_events={
            "clean_optical": clean_cache,
            "particle_source": particle_cache,
            "camera_visibility": visibility_cache,
            "phi_sampling_localization": phi_localization_cache,
        },
        max_workers=max_workers,
    )
    selected_root = resolve_output_root(job, output_root)
    started = perf_counter()
    write_result = write_sequence_roles(
        output_root=selected_root,
        sequence_id=job.sequence_id,
        angles_deg=[pose.angle_deg for pose in job.camera.poses],
        clean_frames=[capture.clean for capture in captures],
        particle_frames=[capture.particle for capture in captures],
        manifest=manifest,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        write_anomaly_preview=write_anomaly_preview,
    )
    stage_seconds["write"] = perf_counter() - started

    return GeneratedSequenceResult(
        sequence_id=job.sequence_id,
        output_path=str(write_result.sequence_directory),
        frame_count=len(captures),
        clean_optical_cache_id=job.clean_optical_cache_id,
        particle_source_cache_id=job.particle_source_cache_id,
        stage_seconds=stage_seconds,
        clean_cache=clean_cache,
        particle_cache=particle_cache,
        visibility_cache=visibility_cache,
        phi_localization_cache=phi_localization_cache,
        warnings=tuple(getattr(particle_state, "notes", ()) or ()),
    )
