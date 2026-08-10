"""Workbook validation, job planning, and batch sequence generation.

Turns Excel control rows into typed ``SequenceJob`` records, groups work by
persistent source-cache identity, reconciles on-disk outputs, and runs physics
for missing sequences. Excel is not runtime authority after validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from gummybear.datasets.cache_keys import (
    clean_optical_cache_key,
    clean_optical_cache_key_payload,
    diffusion_settings_provenance,
    particle_source_cache_key,
    particle_source_cache_key_payload,
)
from gummybear.datasets.generation_workbook import (
    M6Workbook,
    WorkbookRow,
    WorkbookValidationError,
    load_generation_workbook,
)
from gummybear.datasets.source_cache import (
    CLEAN_ALLOW_NONFINITE_ARRAYS,
    CLEAN_PAYLOAD_SCHEMA_VERSION,
    CLEAN_REQUIRED_ARRAYS,
    PARTICLE_PAYLOAD_SCHEMA_VERSION,
    PARTICLE_REQUIRED_ARRAYS,
    SourceCacheStore,
)
from gummybear.datasets.text_table import format_text_table, short_hash
from gummybear.geometry import sha256_file
from gummybear.paths import display_path, repo_relative_path

# Legacy flat default. Prefer scenario-local ``<output_root>/_cache``
# (see :func:`default_cache_root_for_output`).
DEFAULT_CACHE_ROOT = Path("data/generated/_cache")
"""Legacy flat cache root; prefer :func:`default_cache_root_for_output`."""


def default_cache_root_for_output(output_root: Path | str | None) -> Path:
    """Return the scenario-local source-cache root for an output root."""
    if output_root is None:
        return DEFAULT_CACHE_ROOT
    return Path(output_root) / "_cache"


M6_5_MANIFEST_SCHEMA_VERSION = "1.6-m6-draft"
M6_5_GENERATOR_VERSION = "m6.5-draft"
RESOLVED_JOB_HASH_ALGORITHM_VERSION = "m6-resolved-job-v3"
DEFAULT_RUNTIME_SETTINGS: dict[str, Any] = {
    "target_elements": 1000,
    "exitance_scale": 10.0,
    "camera_fov_deg": 35.0,
    "n_from": 1.0,
    "source_delta_assignment": "attenuated_chord",
    "surface_interpolation": "tetrahedral_barycentric",
}


class GenerationPlanError(WorkbookValidationError):
    """Raised when cross-sheet references or scientific constraints fail planning."""


@dataclass(frozen=True)
class OpticalSetupConfig:
    """Phantom-medium illumination and transport coefficients for one setup row.

    Attributes:
        optical_setup_id: Stable setup identifier referenced by sequence rows.
        illumination_kind: Illumination model (Phase 2: ``point`` only at runtime).
        light_position_x, light_position_y, light_position_z: Point-light position.
        num_source_rays: Count of surface source rays sampled for the point light.
        source_intensity: Non-negative source scale (``I0`` in dry-run tables).
        mu_s, mu_a: Background scattering coefficient ``mu_s`` and absorption coefficient ``mu_a``.
        refractive_index: Phantom refractive index for refraction and material D.
        source_deposition_method: Tet-interval deposition policy for clean source.
        cache_policy: Workbook hint; persistent reuse is enforced by cache keys.
        source_sheet, source_excel_row: Workbook provenance (excluded from keys).
    """
    optical_setup_id: str
    illumination_kind: str
    light_position_x: float
    light_position_y: float
    light_position_z: float
    num_source_rays: int
    source_intensity: float
    mu_s: float
    mu_a: float
    refractive_index: float
    source_deposition_method: str
    cache_policy: str
    source_sheet: str = "optical_setups"
    source_excel_row: int | None = None

    def as_optical_material(self, *, g: float = 0.0):
        """Return package material config, optionally with diffusion anisotropy ``g``."""
        from gummybear.optics import OpticalMaterialConfig

        return OpticalMaterialConfig(
            n_refractive=float(self.refractive_index),
            mu_absorption=float(self.mu_a),
            mu_scatter=float(self.mu_s),
            g=float(g),
        )

    def diffusion_coefficient(self, *, g: float) -> float:
        """Return D via ``OpticalMaterialConfig.diffusion_coefficient``.

        Optical scattering coefficient ``mu_s``/absorption coefficient ``mu_a`` plus diffusion-setup ``g`` build the material;
        M6 must not recompute D from a local formula.
        """
        return float(self.as_optical_material(g=g).diffusion_coefficient)


@dataclass(frozen=True)
class ParticleSetupConfig:
    """One analytic sphere inclusion referenced by workbook rows.

    Attributes:
        particle_setup_id: Stable particle identifier.
        particle_kind: Geometry kind (runtime: ``sphere`` only).
        center_x, center_y, center_z: Fixed world-space centre (mm).
        radius: Sphere radius (mm).
        mu_s_particle, mu_a_particle: In-particle scattering coefficient ``mu_s`` and absorption coefficient ``mu_a``.
        refractive_index_particle: Particle refractive index (metadata).
        placement_mode: ``fixed`` at runtime; randomized modes affect cache seed.
        seed: Placement RNG seed when ``placement_mode`` is not ``fixed``.
        enabled: Whether the row participates in group resolution.
        source_sheet, source_excel_row: Workbook provenance.
    """
    particle_setup_id: str
    particle_kind: str
    center_x: float
    center_y: float
    center_z: float
    radius: float
    mu_s_particle: float
    mu_a_particle: float
    refractive_index_particle: float
    placement_mode: str
    seed: int | None
    enabled: bool = True
    source_sheet: str = "particles"
    source_excel_row: int | None = None


@dataclass(frozen=True)
class DiffusionSetupConfig:
    """Finite-element method (FEM) diffusion and hybrid-composition settings for one setup row.

    ``g`` and Robin boundary condition/extrapolation live here; absorption coefficient ``mu_a``/scattering coefficient ``mu_s`` come from the
    linked optical setup. Diffusion provenance is recorded on the job but does
    not appear on clean or particle cache keys.

    Attributes:
        diffusion_setup_id: Stable setup identifier.
        g: Diffusion anisotropy factor for ``OpticalMaterialConfig``.
        robin_boundary_model: Robin boundary condition policy label.
        extrapolation_length: Extrapolation length for Robin boundary condition (BC).
        fem_order: Finite-element method (FEM) polynomial order.
        solver_tolerance: Linear solver tolerance.
        alpha_direct: Hybrid scale ``I_total = alpha * I_direct + I_diffuse``.
        source_sheet, source_excel_row: Workbook provenance.
    """
    diffusion_setup_id: str
    g: float
    robin_boundary_model: str
    extrapolation_length: float
    fem_order: int
    solver_tolerance: float
    alpha_direct: float
    source_sheet: str = "diffusion_setups"
    source_excel_row: int | None = None


@dataclass(frozen=True)
class CameraPose:
    """One camera view in an expanded schedule (a frame, not a dataset role).

    Attributes:
        frame_index: Zero-based ordering source of truth for filenames.
        angle_deg: Orbit angle metadata (human-readable; ordering uses index).
        axis: Rotation axis unit vector.
        distance, elevation_deg: Orbit radius and elevation.
        resolution_x, resolution_y: Image width and height in pixels.
        camera_kind: Camera model label (e.g. ``orbit``).
        lateral_offset, z_offset, up_variant: Optional pose variants (Phase 1: none).
    """
    frame_index: int
    angle_deg: float
    axis: tuple[float, float, float]
    distance: float
    elevation_deg: float
    resolution_x: int
    resolution_y: int
    camera_kind: str
    lateral_offset: float | None = None
    z_offset: float | None = None
    up_variant: str | None = None


@dataclass(frozen=True)
class CameraScheduleConfig:
    """Expanded multi-view camera schedule bound to one setup row.

    Attributes:
        camera_schedule_id: Stable schedule identifier.
        schedule_kind: Expansion policy (e.g. ``orbit``).
        num_views: View count ``V`` in dry-run summaries.
        angle_start_deg, angle_stop_deg: Inclusive-style orbit span.
        axis_x, axis_y, axis_z: Shared rotation axis.
        resolution_x, resolution_y: Shared image resolution.
        camera_kind, distance, elevation_deg: Shared orbit camera parameters.
        lateral_offsets, z_offsets, up_variants: Serialized variant specs or ``none``.
        poses: Deterministic per-view poses from :func:`expand_camera_schedule`.
        source_sheet, source_excel_row: Workbook provenance.
    """
    camera_schedule_id: str
    schedule_kind: str
    num_views: int
    angle_start_deg: float
    angle_stop_deg: float
    axis_x: float
    axis_y: float
    axis_z: float
    resolution_x: int
    resolution_y: int
    camera_kind: str
    distance: float
    elevation_deg: float
    lateral_offsets: str | None
    z_offsets: str | None
    up_variants: str | None
    poses: tuple[CameraPose, ...] = ()
    source_sheet: str = "camera_schedules"
    source_excel_row: int | None = None


@dataclass(frozen=True)
class CorruptionSetupConfig:
    """Optional corruption overlay referenced by a sequence (not yet generated).

    Attributes:
        corruption_setup_id: Stable setup identifier.
        corruption_kind: Corruption operator label (``none`` disables generation).
        amplitude, frames, seed, composition_domain: Kind-specific parameters.
        enabled: Whether corruption would run when supported.
        source_sheet, source_excel_row: Workbook provenance.
    """
    corruption_setup_id: str
    corruption_kind: str
    amplitude: float | None
    frames: str | None
    seed: int | None
    composition_domain: str | None
    enabled: bool
    source_sheet: str = "corruptions"
    source_excel_row: int | None = None


@dataclass(frozen=True)
class SequenceJob:
    """Fully resolved unit of synthetic sequence generation.

    One job produces one ordered multi-view sequence under ``output_root``.
    Role images: ``clean`` (no particle), ``particle``/``observed`` (with particle;
    observed equals particle when corruption is off), optional ``anomaly`` preview
    as ``particle - clean``. Split assignment is ``split`` for catalog partitioning
    by ``sequence_id``.

    Attributes:
        sequence_id: Canonical sequence identifier and output directory name.
        split: Train/val/test label for catalog partitioning by ``sequence_id``
            (workbook / ML metadata; not part of ``resolved_job_hash``).
        seed: Workbook sequence seed used for split randomization bookkeeping.
            Not part of ``resolved_job_hash`` or newly written manifests. Not used
            for illumination source-ray sampling (that uses an unspecified RNG seed).
        phantom_id, stl_path, stl_sha256: Phantom identity (STL triangle mesh file hash in cache keys).
        forward_model_tier: Declared physics tier label.
        optical, particle, particles, diffusion, camera, corruption: Typed setups.
        particles: Ordered group (workbook row order is scientific).
        particle: First particle; compatibility alias for single-particle paths.
        output_root: Scenario directory for sequence folders.
        notes, enabled: Workbook metadata.
        workbook_path, workbook_sha256: Control workbook provenance.
        particle_group_id: Group id when multi-particle; else primary setup id.
        source_excel_row: Sequences-sheet row for manifest coordinates.
        clean_optical_cache_id, particle_source_cache_id: Persistent cache keys.
        clean_optical_cache_payload, particle_source_cache_payload: Key payloads.
        diffusion_provenance: Derived D/mu_a and finite-element method (FEM) settings (not a cache key).
        image_format, jpeg_quality, write_anomaly_preview: Output representation (display JPEG preview vs float sidecar).
        runtime_settings: Extra knobs recorded in ``resolved_job`` (not in keys).
        resolved_job_payload, resolved_job_hash: Output reconciliation identity.

    See also:
        :class:`GenerationPlan` — validated job list from a workbook.
        :func:`~tomography_ml.gummybear_data_catalog.catalog.build_catalog_row` — flat ML catalog row.
    """
    sequence_id: str
    split: str
    seed: int
    phantom_id: str
    stl_path: str
    stl_sha256: str
    forward_model_tier: str
    optical: OpticalSetupConfig
    particle: ParticleSetupConfig
    particles: tuple[ParticleSetupConfig, ...]
    diffusion: DiffusionSetupConfig
    camera: CameraScheduleConfig
    corruption: CorruptionSetupConfig
    output_root: str
    notes: str | None
    enabled: bool
    workbook_path: str
    workbook_sha256: str
    particle_group_id: str = ""
    source_excel_row: int | None = None
    clean_optical_cache_id: str = ""
    particle_source_cache_id: str = ""
    clean_optical_cache_payload: dict[str, Any] = field(default_factory=dict)
    particle_source_cache_payload: dict[str, Any] = field(default_factory=dict)
    diffusion_provenance: dict[str, Any] = field(default_factory=dict)
    image_format: str = "jpg"
    jpeg_quality: int = 95
    write_anomaly_preview: bool = True
    runtime_settings: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_RUNTIME_SETTINGS)
    )
    resolved_job_payload: dict[str, Any] = field(default_factory=dict)
    resolved_job_hash: str = ""


@dataclass(frozen=True)
class GenerationPlan:
    """Validated enabled jobs from one workbook (no physics executed).

    Attributes:
        workbook_path, workbook_sha256: Source workbook identity.
        jobs: Enabled sequences with cache ids and resolved output identity.
        disabled_sequence_ids: Sequence ids with ``enabled=False`` in workbook.
        warnings: Non-fatal planning notices (e.g. placeholder particle centres).

    See also:
        :func:`~gummybear.datasets.generation_workbook.load_generation_workbook` — Excel control input.
        :func:`~tomography_ml.gummybear_data_catalog.gummybear_adapter.load_catalog_jobs` — adapter to ML catalog jobs.
    """
    workbook_path: str
    workbook_sha256: str
    jobs: tuple[SequenceJob, ...]
    disabled_sequence_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def __repr__(self) -> str:
        """Readable summary with repo-relative paths (no absolute home paths)."""
        workbook = display_path(self.workbook_path)
        headers = (
            "sequence_id",
            "split",
            "seed",
            "optical",
            "mu_a",
            "mu_s",
            "I0",
            "particle",
            "n_p",
            "diffusion",
            "camera",
            "V",
            "output_root",
        )
        rows: list[tuple[str, ...]] = []
        for job in self.jobs:
            particle_label = job.particle_group_id or job.particle.particle_setup_id
            rows.append(
                (
                    job.sequence_id,
                    job.split,
                    str(job.seed),
                    job.optical.optical_setup_id,
                    f"{job.optical.mu_a:g}",
                    f"{job.optical.mu_s:g}",
                    f"{job.optical.source_intensity:g}",
                    particle_label,
                    str(len(job.particles)),
                    job.diffusion.diffusion_setup_id,
                    job.camera.camera_schedule_id,
                    str(job.camera.num_views),
                    display_path(job.output_root),
                )
            )

        lines = [
            "GenerationPlan(",
            f"  workbook_path={workbook!r},",
            f"  workbook_sha256={short_hash(self.workbook_sha256)!r},",
            f"  enabled_jobs={len(self.jobs)},",
            f"  disabled_sequence_ids={list(self.disabled_sequence_ids)!r},",
            f"  warnings={len(self.warnings)},",
        ]
        if rows:
            lines.append("  jobs=")
            table = format_text_table(headers, rows)
            lines.extend(f"    {line}" for line in table.splitlines())
        else:
            lines.append("  jobs=(empty),")
        if self.warnings:
            lines.append("  warning_messages=")
            for warning in self.warnings:
                lines.append(f"    - {warning}")
        lines.append(")")
        return "\n".join(lines)


@dataclass(frozen=True)
class CameraTask:
    """One frame capture task in an execution grouping.

    Attributes:
        sequence_id: Owning sequence.
        frame_index, angle_deg: View ordering and metadata.
        resolution_x, resolution_y: Capture resolution for the frame.
    """
    sequence_id: str
    frame_index: int
    angle_deg: float
    resolution_x: int
    resolution_y: int


@dataclass(frozen=True)
class DiffusionGroup:
    """Jobs sharing one diffusion solve configuration after particle grouping.

    Attributes:
        diffusion_setup_id: Workbook diffusion setup id.
        provenance: Effective D, mu_a, Robin boundary condition, and finite-element method (FEM) parameters.
        jobs: Sequences sharing this diffusion configuration.
        camera_tasks: Flattened frame tasks across ``jobs``.
    """
    diffusion_setup_id: str
    provenance: dict[str, Any]
    jobs: tuple[SequenceJob, ...]
    camera_tasks: tuple[CameraTask, ...]


@dataclass(frozen=True)
class ParticleGroup:
    """Jobs sharing one particle-source cache entry under a clean optical group.

    Attributes:
        particle_source_cache_id: SHA256 particle-source cache key.
        particle_setup_id: Primary particle setup id (first of group).
        cache_status: ``hit``, ``miss``, or probe reason from dry-run.
        diffusion_groups: Further grouped by diffusion provenance.
        particle_group_id: Workbook group id when N>1.
        particle_count: Number of spheres in the ordered group.
    """
    particle_source_cache_id: str
    particle_setup_id: str
    cache_status: str
    diffusion_groups: tuple[DiffusionGroup, ...]
    particle_group_id: str = ""
    particle_count: int = 1


@dataclass(frozen=True)
class CleanGroup:
    """Jobs sharing one clean optical source cache.

    Attributes:
        clean_optical_cache_id: SHA256 clean-optical cache key.
        optical_setup_id: Workbook optical setup id.
        cache_status: ``hit``, ``miss``, or probe reason from dry-run.
        particle_groups: Particle-source subgroups under this clean state.
    """
    clean_optical_cache_id: str
    optical_setup_id: str
    cache_status: str
    particle_groups: tuple[ParticleGroup, ...]


@dataclass(frozen=True)
class ExecutionPlan:
    """Dry-run grouping of jobs by shared caches (no physics executed).

    Pipeline: ``GenerationPlan`` → filter/limit → optional output reconcile →
    group by ``clean_optical_cache_id`` → ``particle_source_cache_id`` →
    diffusion provenance → per-frame ``CameraTask`` list.

    Attributes:
        workbook_path, workbook_sha256: Source workbook identity.
        jobs: Selected enabled jobs (after limit/sequence_id filter).
        clean_groups: Hierarchical cache grouping for execution estimates.
        cache_root: Directory holding ``clean_optical/`` and ``particle_source/`` pairs.
        disabled_sequence_ids: Disabled ids carried from the generation plan.
        output_items: Optional ``OutputDeltaItem`` rows when reconciling outputs.
        warnings: Planning warnings propagated from the workbook/plan.
        plans_operator_cache: Whether a diffusion-operator cache is planned (False).
    """
    workbook_path: str
    workbook_sha256: str
    jobs: tuple[SequenceJob, ...]
    clean_groups: tuple[CleanGroup, ...]
    cache_root: str
    disabled_sequence_ids: tuple[str, ...] = ()
    output_items: tuple[Any, ...] = ()
    warnings: tuple[str, ...] = ()
    plans_operator_cache: bool = False


@dataclass(frozen=True)
class DryRunSummary:
    """Aggregate counts for notebook or CLI dry-runs before generation.

    Attributes:
        workbook_path, workbook_sha256: Source workbook identity.
        enabled_sequence_count, disabled_sequence_count: Row counts.
        clean_group_count, particle_group_count, diffusion_group_count: Cache groups.
        sequence_count: Same as enabled sequence count after filters.
        frame_count: Total camera frames across diffusion groups.
        expected_clean_cache_hits, expected_clean_cache_misses: Clean cache probes.
        expected_particle_cache_hits, expected_particle_cache_misses: Particle probes.
        output_roots: Distinct scenario output roots among selected jobs.
        resolutions: Distinct ``(width, height)`` pairs.
        plans_diffusion_operator_cache: Whether operator caching is planned.
        output_status_counts: Counts by ``OutputDeltaItem.status`` when reconciling.
        warnings: Non-fatal planning notices.
    """
    workbook_path: str
    workbook_sha256: str
    enabled_sequence_count: int
    disabled_sequence_count: int
    clean_group_count: int
    particle_group_count: int
    diffusion_group_count: int
    sequence_count: int
    frame_count: int
    expected_clean_cache_hits: int
    expected_clean_cache_misses: int
    expected_particle_cache_hits: int
    expected_particle_cache_misses: int
    output_roots: tuple[str, ...]
    resolutions: tuple[tuple[int, int], ...]
    plans_diffusion_operator_cache: bool
    output_status_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON file-friendly summary with repo-relative ``workbook_path``."""
        data = asdict(self)
        data["workbook_path"] = repo_relative_path(self.workbook_path)
        return data


def _require_value(
    row: WorkbookRow,
    column: str,
    *,
    cast=None,
):
    value = row.values.get(column)
    if value is None:
        raise GenerationPlanError(
            f"Missing required value {column!r} "
            f"(sheet={row.sheet!r} excel_row={row.excel_row} "
            f"setup_id={row.setup_id!r})"
        )
    if cast is None:
        return value
    try:
        return cast(value)
    except Exception as exc:
        raise GenerationPlanError(
            f"Invalid value for {column!r}={value!r} "
            f"(sheet={row.sheet!r} excel_row={row.excel_row} "
            f"setup_id={row.setup_id!r})"
        ) from exc


def _optional_float(value: Any) -> float | None:
    if value is None or value == "none":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "none":
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text


def _none_like(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "none"}


def expand_camera_schedule(
    *,
    camera_schedule_id: str,
    schedule_kind: str,
    num_views: int,
    angle_start_deg: float,
    angle_stop_deg: float,
    axis_x: float,
    axis_y: float,
    axis_z: float,
    resolution_x: int,
    resolution_y: int,
    camera_kind: str,
    distance: float,
    elevation_deg: float,
    lateral_offsets: str | None,
    z_offsets: str | None,
    up_variants: str | None,
    source_excel_row: int | None = None,
) -> CameraScheduleConfig:
    """Expand a simple orbit camera schedule deterministically.

    For ``num_views`` views from ``angle_start_deg`` to ``angle_stop_deg``
    inclusive-style spacing uses:

        angle_i = start + i * (stop - start) / max(num_views - 1, 1)

    when ``num_views > 1``. Offsets/variants equal to ``none`` produce a
    single default pose family.

    Args:
        camera_schedule_id: Stable schedule identifier.
        schedule_kind: Expansion policy label stored on the config.
        num_views: Number of views (must be >= 1).
        angle_start_deg, angle_stop_deg: Orbit angle endpoints in degrees.
        axis_x, axis_y, axis_z: Shared rotation axis components.
        resolution_x, resolution_y: Image size in pixels.
        camera_kind, distance, elevation_deg: Orbit camera parameters.
        lateral_offsets, z_offsets, up_variants: Variant specs or ``none``.
        source_excel_row: Optional workbook row for provenance.

    Returns:
        CameraScheduleConfig with ``poses`` filled in deterministic order.

    Raises:
        GenerationPlanError: Invalid view count, resolution, or unsupported variants.
    """
    if num_views < 1:
        raise GenerationPlanError(
            f"camera_schedule_id={camera_schedule_id!r} has num_views={num_views}"
        )
    if resolution_x < 1 or resolution_y < 1:
        raise GenerationPlanError(
            f"camera_schedule_id={camera_schedule_id!r} has invalid resolution "
            f"{resolution_x}x{resolution_y}"
        )
    if (
        not _none_like(lateral_offsets)
        or not _none_like(z_offsets)
        or not _none_like(up_variants)
    ):
        raise GenerationPlanError(
            f"camera_schedule_id={camera_schedule_id!r}: Phase 1 only supports "
            "lateral_offsets/z_offsets/up_variants = none"
        )

    if num_views == 1:
        angles = [float(angle_start_deg)]
    else:
        span = float(angle_stop_deg) - float(angle_start_deg)
        step = span / float(num_views - 1)
        angles = [float(angle_start_deg) + i * step for i in range(num_views)]

    poses = tuple(
        CameraPose(
            frame_index=i,
            angle_deg=angle,
            axis=(float(axis_x), float(axis_y), float(axis_z)),
            distance=float(distance),
            elevation_deg=float(elevation_deg),
            resolution_x=int(resolution_x),
            resolution_y=int(resolution_y),
            camera_kind=str(camera_kind),
            lateral_offset=None,
            z_offset=None,
            up_variant=None,
        )
        for i, angle in enumerate(angles)
    )

    return CameraScheduleConfig(
        camera_schedule_id=str(camera_schedule_id),
        schedule_kind=str(schedule_kind),
        num_views=int(num_views),
        angle_start_deg=float(angle_start_deg),
        angle_stop_deg=float(angle_stop_deg),
        axis_x=float(axis_x),
        axis_y=float(axis_y),
        axis_z=float(axis_z),
        resolution_x=int(resolution_x),
        resolution_y=int(resolution_y),
        camera_kind=str(camera_kind),
        distance=float(distance),
        elevation_deg=float(elevation_deg),
        lateral_offsets="none" if _none_like(lateral_offsets) else str(lateral_offsets),
        z_offsets="none" if _none_like(z_offsets) else str(z_offsets),
        up_variants="none" if _none_like(up_variants) else str(up_variants),
        poses=poses,
        source_excel_row=source_excel_row,
    )


def _index_setup_rows(
    workbook: M6Workbook,
    sheet: str,
    *,
    require_enabled: bool = False,
) -> dict[str, WorkbookRow]:
    indexed: dict[str, WorkbookRow] = {}
    for row in workbook.rows(sheet):
        setup_id = row.setup_id
        if setup_id is None:
            continue
        if setup_id in indexed:
            raise GenerationPlanError(
                f"Duplicate {SETUP_ID_LABEL[sheet]}={setup_id!r} in sheet "
                f"{sheet!r} (excel_rows={indexed[setup_id].excel_row}, "
                f"{row.excel_row})"
            )
        if require_enabled and not row.enabled:
            continue
        indexed[setup_id] = row
    return indexed


SETUP_ID_LABEL = {
    "sequences": "sequence_id",
    "optical_setups": "optical_setup_id",
    "particles": "particle_setup_id",
    "diffusion_setups": "diffusion_setup_id",
    "camera_schedules": "camera_schedule_id",
    "corruptions": "corruption_setup_id",
}


def _parse_optical(row: WorkbookRow) -> OpticalSetupConfig:
    source_intensity = float(_require_value(row, "source_intensity", cast=float))
    if not math.isfinite(source_intensity) or source_intensity < 0.0:
        raise GenerationPlanError(
            "optical_setups source_intensity must be finite and >= 0 "
            f"(got {source_intensity!r} at excel_row={row.excel_row})."
        )
    return OpticalSetupConfig(
        optical_setup_id=str(_require_value(row, "optical_setup_id")),
        illumination_kind=str(_require_value(row, "illumination_kind")),
        light_position_x=float(_require_value(row, "light_position_x", cast=float)),
        light_position_y=float(_require_value(row, "light_position_y", cast=float)),
        light_position_z=float(_require_value(row, "light_position_z", cast=float)),
        num_source_rays=int(_require_value(row, "num_source_rays", cast=int)),
        source_intensity=source_intensity,
        mu_s=float(_require_value(row, "mu_s", cast=float)),
        mu_a=float(_require_value(row, "mu_a", cast=float)),
        refractive_index=float(_require_value(row, "refractive_index", cast=float)),
        source_deposition_method=str(_require_value(row, "source_deposition_method")),
        cache_policy=str(_require_value(row, "cache_policy")),
        source_excel_row=row.excel_row,
    )


def _parse_particle(row: WorkbookRow) -> ParticleSetupConfig:
    return ParticleSetupConfig(
        particle_setup_id=str(_require_value(row, "particle_setup_id")),
        particle_kind=str(_require_value(row, "particle_kind")),
        center_x=float(_require_value(row, "center_x", cast=float)),
        center_y=float(_require_value(row, "center_y", cast=float)),
        center_z=float(_require_value(row, "center_z", cast=float)),
        radius=float(_require_value(row, "radius", cast=float)),
        mu_s_particle=float(_require_value(row, "mu_s_particle", cast=float)),
        mu_a_particle=float(_require_value(row, "mu_a_particle", cast=float)),
        refractive_index_particle=float(
            _require_value(row, "refractive_index_particle", cast=float)
        ),
        placement_mode=str(_require_value(row, "placement_mode")),
        seed=_optional_int(row.values.get("seed")),
        enabled=bool(row.enabled),
        source_excel_row=row.excel_row,
    )


def _resolve_particle_group(
    sequence_row: WorkbookRow,
    *,
    sequence_id: str,
    particle_rows: dict[str, WorkbookRow],
    particle_sheet_rows: tuple[WorkbookRow, ...],
) -> tuple[str, tuple[ParticleSetupConfig, ...]]:
    """Resolve ordered particles for one sequence (workbook order is scientific).

    If ``particle_group_id`` is set on the sequence row, collect enabled
    ``particles`` sheet rows sharing that group id. Otherwise treat
    ``particle_setup_id`` as a singleton group.
    """
    group_id = _optional_str(sequence_row.values.get("particle_group_id"))
    setup_id = str(_require_value(sequence_row, "particle_setup_id"))

    if group_id is not None:
        member_rows = tuple(
            row
            for row in particle_sheet_rows
            if row.enabled
            and _optional_str(row.values.get("particle_group_id")) == group_id
        )
        if not member_rows:
            raise GenerationPlanError(
                f"sequence_id={sequence_id!r} references empty "
                f"particle_group_id={group_id!r} "
                f"(sheet='sequences' excel_row={sequence_row.excel_row})"
            )
        member_ids = {str(row.values.get("particle_setup_id")) for row in member_rows}
        if setup_id not in member_ids:
            raise GenerationPlanError(
                f"sequence_id={sequence_id!r} particle_setup_id={setup_id!r} "
                f"is not a member of particle_group_id={group_id!r} "
                f"(sheet='sequences' excel_row={sequence_row.excel_row})"
            )
        particles = tuple(_parse_particle(row) for row in member_rows)
        return group_id, particles

    if setup_id not in particle_rows:
        raise GenerationPlanError(
            f"sequence_id={sequence_id!r} references missing "
            f"particle_setup_id={setup_id!r} "
            f"(sheet='sequences' excel_row={sequence_row.excel_row})"
        )
    if not particle_rows[setup_id].enabled:
        raise GenerationPlanError(
            f"sequence_id={sequence_id!r} references disabled "
            f"particle_setup_id={setup_id!r} "
            f"(sheet='sequences' excel_row={sequence_row.excel_row})"
        )
    particle = _parse_particle(particle_rows[setup_id])
    return setup_id, (particle,)


def _validate_fixed_sphere_particles(
    *,
    sequence_id: str,
    particles: tuple[ParticleSetupConfig, ...],
    warnings: list[str],
) -> None:
    from gummybear.particles import ParticleOverlapError, ParticleSet, ParticleSphere

    if not particles:
        raise GenerationPlanError(
            f"sequence_id={sequence_id!r}: particle group is empty."
        )
    for particle in particles:
        if particle.particle_kind != "sphere":
            raise GenerationPlanError(
                f"sequence_id={sequence_id!r}: v1 multi-particle jobs require "
                f"particle_kind='sphere' "
                f"(got {particle.particle_kind!r} for "
                f"{particle.particle_setup_id!r})"
            )
        if particle.placement_mode != "fixed":
            raise GenerationPlanError(
                f"sequence_id={sequence_id!r}: v1 multi-particle jobs require "
                f"placement_mode='fixed' "
                f"(got {particle.placement_mode!r} for "
                f"{particle.particle_setup_id!r})"
            )
        if (
            particle.center_x == 0.0
            and particle.center_y == 0.0
            and particle.center_z == 0.0
        ):
            warnings.append(
                f"sequence_id={sequence_id!r}: particle "
                f"{particle.particle_setup_id!r} center is a Phase 1 "
                "smoke placeholder; later phases should reuse a validated "
                "M5D in-mesh placement."
            )

    try:
        ParticleSet.from_particles(
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
                for item in particles
            ]
        )
    except ParticleOverlapError as exc:
        raise GenerationPlanError(
            f"sequence_id={sequence_id!r}: {exc}"
        ) from exc


def _parse_diffusion(row: WorkbookRow) -> DiffusionSetupConfig:
    return DiffusionSetupConfig(
        diffusion_setup_id=str(_require_value(row, "diffusion_setup_id")),
        g=float(_require_value(row, "g", cast=float)),
        robin_boundary_model=str(_require_value(row, "robin_boundary_model")),
        extrapolation_length=float(
            _require_value(row, "extrapolation_length", cast=float)
        ),
        fem_order=int(_require_value(row, "fem_order", cast=int)),
        solver_tolerance=float(_require_value(row, "solver_tolerance", cast=float)),
        alpha_direct=float(_require_value(row, "alpha_direct", cast=float)),
        source_excel_row=row.excel_row,
    )


def _parse_camera(row: WorkbookRow) -> CameraScheduleConfig:
    return expand_camera_schedule(
        camera_schedule_id=str(_require_value(row, "camera_schedule_id")),
        schedule_kind=str(_require_value(row, "schedule_kind")),
        num_views=int(_require_value(row, "num_views", cast=int)),
        angle_start_deg=float(_require_value(row, "angle_start_deg", cast=float)),
        angle_stop_deg=float(_require_value(row, "angle_stop_deg", cast=float)),
        axis_x=float(_require_value(row, "axis_x", cast=float)),
        axis_y=float(_require_value(row, "axis_y", cast=float)),
        axis_z=float(_require_value(row, "axis_z", cast=float)),
        resolution_x=int(_require_value(row, "resolution_x", cast=int)),
        resolution_y=int(_require_value(row, "resolution_y", cast=int)),
        camera_kind=str(_require_value(row, "camera_kind")),
        distance=float(_require_value(row, "distance", cast=float)),
        elevation_deg=float(_require_value(row, "elevation_deg", cast=float)),
        lateral_offsets=_optional_str(row.values.get("lateral_offsets")),
        z_offsets=_optional_str(row.values.get("z_offsets")),
        up_variants=_optional_str(row.values.get("up_variants")),
        source_excel_row=row.excel_row,
    )


def _parse_corruption(row: WorkbookRow) -> CorruptionSetupConfig:
    return CorruptionSetupConfig(
        corruption_setup_id=str(_require_value(row, "corruption_setup_id")),
        corruption_kind=str(_require_value(row, "corruption_kind")),
        amplitude=_optional_float(row.values.get("amplitude")),
        frames=_optional_str(row.values.get("frames")),
        seed=_optional_int(row.values.get("seed")),
        composition_domain=_optional_str(row.values.get("composition_domain")),
        enabled=bool(row.enabled),
        source_excel_row=row.excel_row,
    )


def _resolve_stl_sha256(
    stl_path: str,
    *,
    repo_root: Path | None = None,
    stl_root: Path | None = None,
) -> str:
    """Hash an STL triangle mesh file for cache identity.

    ``stl_path`` remains the workbook-relative (or absolute) string used in
    scientific identity. ``stl_root`` / ``repo_root`` only locate the STL mesh file on
    disk for content hashing when ``stl_path`` is relative.
    """
    path = Path(stl_path)
    candidates = [path]
    if not path.is_absolute():
        if stl_root is not None:
            candidates.append(Path(stl_root) / path)
        if repo_root is not None:
            candidates.append(Path(repo_root) / path)
        candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.is_file():
            return sha256_file(candidate)

    raise GenerationPlanError(
        f"STL file not found for hashing: {stl_path!r}. "
        "Phase 1 requires cad/proto_bear.stl (or the configured stl_path) "
        "to exist so the clean optical cache key can include stl_sha256."
        + (
            f" Tried stl_root={str(stl_root)!r}."
            if stl_root is not None
            else ""
        )
    )


def _scientific_setup_payload(setup: Any) -> dict[str, Any]:
    """Return setup values without workbook-location provenance."""
    return {
        key: value
        for key, value in asdict(setup).items()
        if key not in {"source_sheet", "source_excel_row"}
    }


def _particle_setup_cache_mapping(setup: ParticleSetupConfig) -> dict[str, Any]:
    """Map one ParticleSetupConfig to particle-source cache-key fields."""
    return {
        "particle_setup_id": setup.particle_setup_id,
        "particle_kind": setup.particle_kind,
        "center_x": setup.center_x,
        "center_y": setup.center_y,
        "center_z": setup.center_z,
        "radius": setup.radius,
        "mu_s_particle": setup.mu_s_particle,
        "mu_a_particle": setup.mu_a_particle,
        "refractive_index_particle": setup.refractive_index_particle,
    }


def _particles_manifest_block(
    particles: tuple[ParticleSetupConfig, ...],
    *,
    particle_group_id: str,
) -> dict[str, Any]:
    """Ordered particles block for manifests / resolved-job identity."""
    return {
        "particle_group_id": particle_group_id,
        "count": len(particles),
        "order": "workbook_row_order",
        "items": [_scientific_setup_payload(item) for item in particles],
    }


def resolved_job_identity_payload(
    job: SequenceJob,
    *,
    image_format: str | None = None,
    jpeg_quality: int | None = None,
    write_anomaly_preview: bool | None = None,
    runtime_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical per-sequence scientific and output identity payload.

    Hashes of this payload become ``resolved_job_hash`` for output reconciliation.
    Workbook coordinates, train/validation/test ``split``, and workbook
    ``sequences.seed`` are excluded; cache key payloads and representation
    contracts are included.

    Args:
        job: Fully planned sequence job with cache ids attached.
        image_format: Override display JPEG preview format label (default: job value).
        jpeg_quality: Override display JPEG preview quality 1–100 (default: job value).
        write_anomaly_preview: Override anomaly PNG sidecar flag.
        runtime_settings: Override recorded runtime knobs (not cache keys).

    Returns:
        JSON-serializable identity dict including setups, caches, and roles.
    """
    selected_format = (
        job.image_format if image_format is None else image_format.strip().lower()
    )
    if selected_format == "jpeg":
        selected_format = "jpg"
    selected_quality = job.jpeg_quality if jpeg_quality is None else int(jpeg_quality)
    selected_anomaly = (
        job.write_anomaly_preview
        if write_anomaly_preview is None
        else bool(write_anomaly_preview)
    )
    selected_runtime = dict(
        job.runtime_settings if runtime_settings is None else runtime_settings
    )
    return {
        "hash_algorithm_version": RESOLVED_JOB_HASH_ALGORITHM_VERSION,
        "manifest_schema_version": M6_5_MANIFEST_SCHEMA_VERSION,
        "generator_version": M6_5_GENERATOR_VERSION,
        "sequence": {
            "sequence_id": job.sequence_id,
            "phantom_id": job.phantom_id,
            "stl_sha256": job.stl_sha256,
            "forward_model_tier": job.forward_model_tier,
        },
        "setups": {
            "optical": _scientific_setup_payload(job.optical),
            "particle": _scientific_setup_payload(job.particle),
            "particles": _particles_manifest_block(
                job.particles,
                particle_group_id=job.particle_group_id,
            ),
            "diffusion": _scientific_setup_payload(job.diffusion),
            "diffusion_effective": dict(job.diffusion_provenance),
            "camera": _scientific_setup_payload(job.camera),
            "corruption": _scientific_setup_payload(job.corruption),
        },
        "source_cache_identity": {
            "clean_optical_cache_id": job.clean_optical_cache_id,
            "particle_source_cache_id": job.particle_source_cache_id,
            "clean_key_algorithm_version": job.clean_optical_cache_payload.get(
                "algorithm_version"
            ),
            "particle_key_algorithm_version": job.particle_source_cache_payload.get(
                "algorithm_version"
            ),
        },
        "representation": {
            "image_format": selected_format,
            "jpeg_quality": selected_quality,
            "image_domain": "camera_intensity",
            "composition_domain": "linear_camera_intensity_before_jpeg",
            "anomaly_definition": "particle_minus_clean",
            "observed_definition": "particle_no_corruption",
            "required_roles": ["clean", "particle", "observed"],
            "write_anomaly_preview": selected_anomaly,
        },
        "runtime_settings": selected_runtime,
    }


def resolve_job_output_identity(
    job: SequenceJob,
    *,
    image_format: str | None = None,
    jpeg_quality: int | None = None,
    write_anomaly_preview: bool | None = None,
    runtime_settings: Mapping[str, Any] | None = None,
) -> SequenceJob:
    """Return ``job`` with deterministic resolved output identity attached.

    Args:
        job: Planned job with cache ids and setups.
        image_format, jpeg_quality, write_anomaly_preview, runtime_settings:
            Optional overrides merged into the identity payload.

    Returns:
        Copy of ``job`` with ``resolved_job_payload`` and ``resolved_job_hash`` set.
    """
    payload = resolved_job_identity_payload(
        job,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        write_anomaly_preview=write_anomaly_preview,
        runtime_settings=runtime_settings,
    )
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    representation = payload["representation"]
    return replace(
        job,
        image_format=str(representation["image_format"]),
        jpeg_quality=int(representation["jpeg_quality"]),
        write_anomaly_preview=bool(representation["write_anomaly_preview"]),
        runtime_settings=dict(payload["runtime_settings"]),
        resolved_job_payload=payload,
        resolved_job_hash=digest,
    )


def _attach_cache_ids(job: SequenceJob) -> SequenceJob:
    clean_payload = clean_optical_cache_key_payload(
        stl_sha256=job.stl_sha256,
        illumination_kind=job.optical.illumination_kind,
        light_position_x=job.optical.light_position_x,
        light_position_y=job.optical.light_position_y,
        light_position_z=job.optical.light_position_z,
        num_source_rays=job.optical.num_source_rays,
        source_intensity=job.optical.source_intensity,
        source_ray_seed=None,
        mu_s=job.optical.mu_s,
        mu_a=job.optical.mu_a,
        refractive_index=job.optical.refractive_index,
        source_deposition_method=job.optical.source_deposition_method,
    )
    clean_id = clean_optical_cache_key(
        stl_sha256=job.stl_sha256,
        illumination_kind=job.optical.illumination_kind,
        light_position_x=job.optical.light_position_x,
        light_position_y=job.optical.light_position_y,
        light_position_z=job.optical.light_position_z,
        num_source_rays=job.optical.num_source_rays,
        source_intensity=job.optical.source_intensity,
        source_ray_seed=None,
        mu_s=job.optical.mu_s,
        mu_a=job.optical.mu_a,
        refractive_index=job.optical.refractive_index,
        source_deposition_method=job.optical.source_deposition_method,
    )

    particle_payload = particle_source_cache_key_payload(
        clean_optical_cache_id=clean_id,
        particles=[
            _particle_setup_cache_mapping(item) for item in job.particles
        ],
        placement_mode=job.particle.placement_mode,
        seed=job.particle.seed,
        source_delta_assignment="attenuated_chord",
    )
    particle_id = particle_source_cache_key(
        clean_optical_cache_id=clean_id,
        particles=[
            _particle_setup_cache_mapping(item) for item in job.particles
        ],
        placement_mode=job.particle.placement_mode,
        seed=job.particle.seed,
        source_delta_assignment="attenuated_chord",
    )

    derived_d = job.optical.diffusion_coefficient(g=job.diffusion.g)
    provenance = diffusion_settings_provenance(
        diffusion_setup_id=job.diffusion.diffusion_setup_id,
        D=derived_d,
        mu_a=job.optical.mu_a,
        robin_boundary_model=job.diffusion.robin_boundary_model,
        extrapolation_length=job.diffusion.extrapolation_length,
        fem_order=job.diffusion.fem_order,
        solver_tolerance=job.diffusion.solver_tolerance,
        alpha_direct=job.diffusion.alpha_direct,
        mu_a_source="optical_setup",
        D_source="optical_material_diffusion_coefficient",
        optical_setup_id=job.optical.optical_setup_id,
        g=job.diffusion.g,
        g_source="diffusion_setup",
    )

    resolved = SequenceJob(
        sequence_id=job.sequence_id,
        split=job.split,
        seed=job.seed,
        phantom_id=job.phantom_id,
        stl_path=job.stl_path,
        stl_sha256=job.stl_sha256,
        forward_model_tier=job.forward_model_tier,
        optical=job.optical,
        particle=job.particle,
        particles=job.particles,
        diffusion=job.diffusion,
        camera=job.camera,
        corruption=job.corruption,
        output_root=job.output_root,
        notes=job.notes,
        enabled=job.enabled,
        workbook_path=job.workbook_path,
        workbook_sha256=job.workbook_sha256,
        particle_group_id=job.particle_group_id,
        source_excel_row=job.source_excel_row,
        clean_optical_cache_id=clean_id,
        particle_source_cache_id=particle_id,
        clean_optical_cache_payload=clean_payload,
        particle_source_cache_payload=particle_payload,
        diffusion_provenance=provenance,
    )
    return resolve_job_output_identity(resolved)


def validate_generation_plan(
    workbook: M6Workbook,
    *,
    repo_root: Path | str | None = None,
    stl_root: Path | str | None = None,
) -> GenerationPlan:
    """Validate workbook cross-references and build typed sequence jobs.

    ``repo_root`` is the default base for relative ``stl_path`` STL triangle mesh file lookup.
    ``stl_root``, when set, is tried first for relative STL paths so catalog /
    generation roots can differ from the CAD tree. The workbook ``stl_path``
    string itself is unchanged and remains the cache-key identity input;
    only the on-disk location used for ``stl_sha256`` changes.
    """
    if repo_root is None:
        # Prefer repository root inferred from workbook path when possible.
        # configs/m6/m6_test.xlsx -> repo root is parent of configs/.
        candidate = workbook.path.parent
        if candidate.name == "configs":
            repo_root_path = candidate.parent
        else:
            repo_root_path = Path.cwd()
    else:
        repo_root_path = Path(repo_root)
    stl_root_path = Path(stl_root) if stl_root is not None else None

    optical_rows = _index_setup_rows(workbook, "optical_setups")
    particle_rows = _index_setup_rows(workbook, "particles")
    diffusion_rows = _index_setup_rows(workbook, "diffusion_setups")
    camera_rows = _index_setup_rows(workbook, "camera_schedules")
    corruption_rows = _index_setup_rows(workbook, "corruptions")

    disabled_sequence_ids: list[str] = []
    enabled_ids: set[str] = set()
    jobs: list[SequenceJob] = []
    warnings: list[str] = list(workbook.warnings)

    for row in workbook.rows("sequences"):
        sequence_id = str(_require_value(row, "sequence_id"))
        if not row.enabled:
            disabled_sequence_ids.append(sequence_id)
            continue
        if sequence_id in enabled_ids:
            raise GenerationPlanError(
                f"Duplicate enabled sequence_id={sequence_id!r} "
                f"(sheet='sequences' excel_row={row.excel_row})"
            )
        enabled_ids.add(sequence_id)

        optical_id = str(_require_value(row, "optical_setup_id"))
        particle_id = str(_require_value(row, "particle_setup_id"))
        diffusion_id = str(_require_value(row, "diffusion_setup_id"))
        camera_id = str(_require_value(row, "camera_schedule_id"))
        corruption_id = str(_require_value(row, "corruption_setup_id"))

        if optical_id not in optical_rows:
            raise GenerationPlanError(
                f"sequence_id={sequence_id!r} references missing "
                f"optical_setup_id={optical_id!r} "
                f"(sheet='sequences' excel_row={row.excel_row})"
            )
        if diffusion_id not in diffusion_rows:
            raise GenerationPlanError(
                f"sequence_id={sequence_id!r} references missing "
                f"diffusion_setup_id={diffusion_id!r} "
                f"(sheet='sequences' excel_row={row.excel_row})"
            )
        if camera_id not in camera_rows:
            raise GenerationPlanError(
                f"sequence_id={sequence_id!r} references missing "
                f"camera_schedule_id={camera_id!r} "
                f"(sheet='sequences' excel_row={row.excel_row})"
            )
        if corruption_id not in corruption_rows:
            raise GenerationPlanError(
                f"sequence_id={sequence_id!r} references missing "
                f"corruption_setup_id={corruption_id!r} "
                f"(sheet='sequences' excel_row={row.excel_row})"
            )

        stl_path = str(_require_value(row, "stl_path"))
        stl_sha256 = _resolve_stl_sha256(
            stl_path,
            repo_root=repo_root_path,
            stl_root=stl_root_path,
        )

        optical = _parse_optical(optical_rows[optical_id])
        particle_group_id, particles = _resolve_particle_group(
            row,
            sequence_id=sequence_id,
            particle_rows=particle_rows,
            particle_sheet_rows=workbook.rows("particles"),
        )
        _validate_fixed_sphere_particles(
            sequence_id=sequence_id,
            particles=particles,
            warnings=warnings,
        )
        particle = particles[0]
        diffusion = _parse_diffusion(diffusion_rows[diffusion_id])
        camera = _parse_camera(camera_rows[camera_id])
        corruption = _parse_corruption(corruption_rows[corruption_id])

        job = SequenceJob(
            sequence_id=sequence_id,
            split=str(_require_value(row, "split")),
            seed=int(_require_value(row, "seed", cast=int)),
            phantom_id=str(_require_value(row, "phantom_id")),
            stl_path=stl_path,
            stl_sha256=stl_sha256,
            forward_model_tier=str(_require_value(row, "forward_model_tier")),
            optical=optical,
            particle=particle,
            particles=particles,
            diffusion=diffusion,
            camera=camera,
            corruption=corruption,
            output_root=str(_require_value(row, "output_root")),
            notes=_optional_str(row.values.get("notes")),
            enabled=True,
            workbook_path=str(workbook.path),
            workbook_sha256=workbook.sha256,
            particle_group_id=particle_group_id,
            source_excel_row=row.excel_row,
        )
        jobs.append(_attach_cache_ids(job))

    return GenerationPlan(
        workbook_path=str(workbook.path),
        workbook_sha256=workbook.sha256,
        jobs=tuple(jobs),
        disabled_sequence_ids=tuple(disabled_sequence_ids),
        warnings=tuple(warnings),
    )


def _probe_cache_status(
    *,
    cache_root: Path,
    kind: str,
    cache_id: str,
    key_payload: dict[str, Any],
    payload_schema_version: str,
    required_arrays: tuple[str, ...],
    allow_nonfinite_arrays: tuple[str, ...] = (),
) -> str:
    """Validate a completed cache pair without requiring live mesh state."""
    result = SourceCacheStore(cache_root).load(
        kind=kind,
        cache_id=cache_id,
        key_payload=key_payload,
        payload_schema_version=payload_schema_version,
        required_arrays=required_arrays,
        mesh_identity=None,
        allow_nonfinite_arrays=allow_nonfinite_arrays,
    )
    return result.event.status


def build_execution_plan(
    plan: GenerationPlan,
    *,
    limit: int | None = None,
    sequence_id: str | None = None,
    cache_root: Path | str | None = None,
    output_root: Path | str | None = None,
    reconcile_outputs: bool = False,
) -> ExecutionPlan:
    """Group validated jobs into a dry-run execution plan (no physics).

    Args:
        plan: Validated generation plan from :func:`validate_generation_plan`.
        limit: Optional maximum number of sequences after sorting by id.
        sequence_id: Optional filter to one enabled sequence.
        cache_root: Override cache directory (else derived from output/jobs).
        output_root: Scenario root used with ``default_cache_root_for_output``.
        reconcile_outputs: When True, classify on-disk outputs and group only
            sequences marked missing.

    Returns:
        ExecutionPlan with hierarchical clean → particle → diffusion groups.
    """
    jobs = list(plan.jobs)

    if sequence_id is not None:
        jobs = [job for job in jobs if job.sequence_id == sequence_id]
        if not jobs:
            raise GenerationPlanError(
                f"No enabled sequence_id={sequence_id!r} in generation plan."
            )

    jobs = sorted(jobs, key=lambda job: job.sequence_id)
    if limit is not None:
        if limit < 0:
            raise GenerationPlanError(f"limit must be >= 0, got {limit}")
        jobs = jobs[:limit]

    selected_jobs = tuple(jobs)
    if cache_root is not None:
        cache_root_path = Path(cache_root)
    elif output_root is not None:
        cache_root_path = default_cache_root_for_output(output_root)
    elif selected_jobs:
        cache_root_path = default_cache_root_for_output(selected_jobs[0].output_root)
    else:
        cache_root_path = DEFAULT_CACHE_ROOT
    output_items: tuple[Any, ...] = ()
    jobs_to_group = list(selected_jobs)
    if reconcile_outputs:
        from gummybear.datasets.output_plan import (
            OUTPUT_MISSING,
            build_output_delta_plan,
        )

        delta = build_output_delta_plan(
            selected_jobs,
            output_root=output_root,
            disabled_sequence_ids=plan.disabled_sequence_ids,
        )
        output_items = delta.requested + delta.disabled + delta.orphaned
        missing_ids = {
            item.sequence_id
            for item in delta.requested
            if item.status == OUTPUT_MISSING
        }
        jobs_to_group = [job for job in selected_jobs if job.sequence_id in missing_ids]

    clean_map: dict[str, list[SequenceJob]] = {}
    for job in jobs_to_group:
        clean_map.setdefault(job.clean_optical_cache_id, []).append(job)

    clean_groups: list[CleanGroup] = []
    for clean_id in sorted(clean_map):
        clean_jobs = clean_map[clean_id]
        optical_setup_id = clean_jobs[0].optical.optical_setup_id
        clean_status = _probe_cache_status(
            cache_root=cache_root_path,
            kind="clean_optical",
            cache_id=clean_id,
            key_payload=clean_jobs[0].clean_optical_cache_payload,
            payload_schema_version=CLEAN_PAYLOAD_SCHEMA_VERSION,
            required_arrays=CLEAN_REQUIRED_ARRAYS,
            allow_nonfinite_arrays=CLEAN_ALLOW_NONFINITE_ARRAYS,
        )

        particle_map: dict[str, list[SequenceJob]] = {}
        for job in clean_jobs:
            particle_map.setdefault(job.particle_source_cache_id, []).append(job)

        particle_groups: list[ParticleGroup] = []
        for particle_id in sorted(particle_map):
            particle_jobs = particle_map[particle_id]
            particle_setup_id = particle_jobs[0].particle.particle_setup_id
            particle_status = _probe_cache_status(
                cache_root=cache_root_path,
                kind="particle_source",
                cache_id=particle_id,
                key_payload=particle_jobs[0].particle_source_cache_payload,
                payload_schema_version=PARTICLE_PAYLOAD_SCHEMA_VERSION,
                required_arrays=PARTICLE_REQUIRED_ARRAYS,
            )

            diffusion_map: dict[str, list[SequenceJob]] = {}
            for job in particle_jobs:
                # Group by setup id plus derived material D/mu_a so diffusion
                # anisotropy (g) cannot silently merge incompatible solves.
                diffusion_key = (
                    job.diffusion.diffusion_setup_id,
                    float(job.diffusion_provenance["D"]),
                    float(job.diffusion_provenance["mu_a"]),
                    float(job.diffusion.g),
                    float(job.diffusion.alpha_direct),
                    float(job.diffusion.extrapolation_length),
                    int(job.diffusion.fem_order),
                    float(job.diffusion.solver_tolerance),
                    str(job.diffusion.robin_boundary_model),
                )
                diffusion_map.setdefault(repr(diffusion_key), []).append(job)

            diffusion_groups: list[DiffusionGroup] = []
            for serialized_key in sorted(diffusion_map):
                diffusion_jobs = tuple(
                    sorted(
                        diffusion_map[serialized_key],
                        key=lambda job: job.sequence_id,
                    )
                )
                camera_tasks: list[CameraTask] = []
                for job in diffusion_jobs:
                    for pose in job.camera.poses:
                        camera_tasks.append(
                            CameraTask(
                                sequence_id=job.sequence_id,
                                frame_index=pose.frame_index,
                                angle_deg=pose.angle_deg,
                                resolution_x=pose.resolution_x,
                                resolution_y=pose.resolution_y,
                            )
                        )
                diffusion_groups.append(
                    DiffusionGroup(
                        diffusion_setup_id=(
                            diffusion_jobs[0].diffusion.diffusion_setup_id
                        ),
                        provenance=dict(diffusion_jobs[0].diffusion_provenance),
                        jobs=diffusion_jobs,
                        camera_tasks=tuple(camera_tasks),
                    )
                )

            particle_groups.append(
                ParticleGroup(
                    particle_source_cache_id=particle_id,
                    particle_setup_id=particle_setup_id,
                    cache_status=particle_status,
                    diffusion_groups=tuple(diffusion_groups),
                    particle_group_id=particle_jobs[0].particle_group_id,
                    particle_count=len(particle_jobs[0].particles),
                )
            )

        clean_groups.append(
            CleanGroup(
                clean_optical_cache_id=clean_id,
                optical_setup_id=optical_setup_id,
                cache_status=clean_status,
                particle_groups=tuple(particle_groups),
            )
        )

    return ExecutionPlan(
        workbook_path=plan.workbook_path,
        workbook_sha256=plan.workbook_sha256,
        jobs=selected_jobs,
        clean_groups=tuple(clean_groups),
        cache_root=str(cache_root_path),
        disabled_sequence_ids=plan.disabled_sequence_ids,
        output_items=output_items,
        warnings=plan.warnings,
        plans_operator_cache=False,
    )


def summarize_execution_plan(
    execution_plan: ExecutionPlan,
    *,
    disabled_sequence_count: int | None = None,
) -> DryRunSummary:
    """Build a notebook-friendly dry-run summary from an execution plan.

    Args:
        execution_plan: Grouped plan from :func:`build_execution_plan`.
        disabled_sequence_count: Override disabled count (default 0).

    Returns:
        DryRunSummary with cache hit/miss estimates and frame totals.
    """
    frame_count = 0
    resolutions: set[tuple[int, int]] = set()
    output_roots: set[str] = set()
    particle_group_count = 0
    diffusion_group_count = 0
    clean_hits = 0
    clean_misses = 0
    particle_hits = 0
    particle_misses = 0
    output_status_counts: dict[str, int] = {}
    for item in execution_plan.output_items:
        status = str(item.status)
        output_status_counts[status] = output_status_counts.get(status, 0) + 1

    for clean_group in execution_plan.clean_groups:
        if clean_group.cache_status == "hit":
            clean_hits += 1
        else:
            clean_misses += 1
        for particle_group in clean_group.particle_groups:
            particle_group_count += 1
            if particle_group.cache_status == "hit":
                particle_hits += 1
            else:
                particle_misses += 1
            for diffusion_group in particle_group.diffusion_groups:
                diffusion_group_count += 1
                frame_count += len(diffusion_group.camera_tasks)
                for job in diffusion_group.jobs:
                    output_roots.add(job.output_root)
                    resolutions.add((job.camera.resolution_x, job.camera.resolution_y))

    return DryRunSummary(
        workbook_path=execution_plan.workbook_path,
        workbook_sha256=execution_plan.workbook_sha256,
        enabled_sequence_count=len(execution_plan.jobs),
        disabled_sequence_count=(
            0 if disabled_sequence_count is None else int(disabled_sequence_count)
        ),
        clean_group_count=len(execution_plan.clean_groups),
        particle_group_count=particle_group_count,
        diffusion_group_count=diffusion_group_count,
        sequence_count=len(execution_plan.jobs),
        frame_count=frame_count,
        expected_clean_cache_hits=clean_hits,
        expected_clean_cache_misses=clean_misses,
        expected_particle_cache_hits=particle_hits,
        expected_particle_cache_misses=particle_misses,
        output_roots=tuple(sorted(output_roots)),
        resolutions=tuple(sorted(resolutions)),
        plans_diffusion_operator_cache=False,
        output_status_counts=output_status_counts,
        warnings=execution_plan.warnings,
    )


def load_and_summarize_generation_workbook(
    path: Path | str,
    *,
    limit: int | None = None,
    sequence_id: str | None = None,
    cache_root: Path | str | None = None,
    output_root: Path | str | None = None,
    reconcile_outputs: bool = False,
    repo_root: Path | str | None = None,
    stl_root: Path | str | None = None,
) -> tuple[M6Workbook, GenerationPlan, ExecutionPlan, DryRunSummary]:
    """Load a workbook and return plan, execution plan, and dry-run summary.

    Args:
        path: Generation workbook ``.xlsx`` path.
        limit, sequence_id, cache_root, output_root, reconcile_outputs:
            Forwarded to :func:`build_execution_plan`.
        repo_root, stl_root: STL lookup bases for :func:`validate_generation_plan`.

    Returns:
        Tuple ``(workbook, generation_plan, execution_plan, dry_run_summary)``.

    Notebook / protocol:
        Primary one-call dry-run entry before :func:`run_generation_workbook`.
    """
    workbook = load_generation_workbook(path)
    plan = validate_generation_plan(
        workbook,
        repo_root=repo_root,
        stl_root=stl_root,
    )
    execution_plan = build_execution_plan(
        plan,
        limit=limit,
        sequence_id=sequence_id,
        cache_root=cache_root,
        output_root=output_root,
        reconcile_outputs=reconcile_outputs,
    )
    summary = summarize_execution_plan(
        execution_plan,
        disabled_sequence_count=len(plan.disabled_sequence_ids),
    )
    return workbook, plan, execution_plan, summary


def default_parallel_workers() -> int:
    """Return a conservative local worker count: available CPUs minus two.

    Always at least 1 so a single-core host still runs. Leaves headroom so the
    machine remains interactive during batch generation.
    """
    return max(1, (os.cpu_count() or 1) - 2)


def resolve_generation_workers(
    *,
    parallel: bool = False,
    max_workers: int | None = None,
) -> int:
    """Resolve the number of sequence-level worker processes/threads.

    Serial baseline (``parallel=False``) always uses 1 worker. Parallel mode
    uses ``max_workers`` when set, otherwise :func:`default_parallel_workers`.
    """
    if not parallel:
        if max_workers is not None and int(max_workers) != 1:
            raise GenerationPlanError(
                "max_workers>1 requires parallel=True "
                "(serial baseline uses one sequence worker)."
            )
        return 1
    if max_workers is None:
        return default_parallel_workers()
    workers = int(max_workers)
    if workers < 1:
        raise GenerationPlanError(f"max_workers must be >= 1, got {workers}")
    return workers


def _generate_smoke_sequence_process(
    job: SequenceJob,
    *,
    output_root: str | None,
    cache_root: str | None,
    force_recompute: bool,
    settings: dict[str, Any],
    image_format: str,
    jpeg_quality: int,
    write_anomaly_preview: bool,
    max_workers: int,
):
    """Picklable process-pool entry that rebuilds the default physics backend."""
    from gummybear.datasets.sequence_generation import (
        SmokeRuntimeSettings,
        generate_smoke_sequence,
    )

    return generate_smoke_sequence(
        job,
        output_root=output_root,
        cache_root=cache_root,
        force_recompute=force_recompute,
        backend=None,
        settings=SmokeRuntimeSettings(**settings),
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        write_anomaly_preview=write_anomaly_preview,
        max_workers=max_workers,
    )


def _report_generation_progress(done: int, total: int, *, enabled: bool) -> None:
    """Print compact ``done/total`` progress when enabled."""
    if enabled:
        print(f"{done}/{total}", flush=True)


def _run_parallel_sequence_batch(
    jobs_to_generate: list[SequenceJob],
    *,
    workers: int,
    output_root: Path | str | None,
    cache_root: Path | str | None,
    force_recompute: bool,
    physics_backend,
    settings,
    image_format: str,
    jpeg_quality: int,
    write_anomaly_preview: bool,
    common_kwargs: dict[str, Any],
    progress: bool = False,
    progress_done: int = 0,
    progress_total: int | None = None,
):
    """Run ``jobs_to_generate`` concurrently (per-sequence only)."""
    from gummybear.datasets.sequence_generation import generate_smoke_sequence

    if not jobs_to_generate:
        return ()
    total = len(jobs_to_generate) if progress_total is None else progress_total
    if len(jobs_to_generate) == 1 or workers == 1:
        generated = []
        done = progress_done
        for job in jobs_to_generate:
            generated.append(
                generate_smoke_sequence(
                    job,
                    backend=physics_backend,
                    **common_kwargs,
                )
            )
            done += 1
            _report_generation_progress(done, total, enabled=progress)
        return tuple(generated)

    worker_count = min(workers, len(jobs_to_generate))
    # Injected backends are typically not picklable; use threads so tests and
    # notebook overrides remain usable. Default physics uses processes so FEM
    # work can occupy distinct cores without sharing NGSolve state.
    if physics_backend is not None:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(
                    generate_smoke_sequence,
                    job,
                    backend=physics_backend,
                    **common_kwargs,
                )
                for job in jobs_to_generate
            ]
            results_by_future = {}
            done = progress_done
            for future in as_completed(futures):
                results_by_future[future] = future.result()
                done += 1
                _report_generation_progress(done, total, enabled=progress)
            return tuple(results_by_future[future] for future in futures)

    output_root_s = None if output_root is None else str(output_root)
    cache_root_s = None if cache_root is None else str(cache_root)
    settings_dict = asdict(settings)
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(
                _generate_smoke_sequence_process,
                job,
                output_root=output_root_s,
                cache_root=cache_root_s,
                force_recompute=force_recompute,
                settings=settings_dict,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
                write_anomaly_preview=write_anomaly_preview,
                max_workers=workers,
            )
            for job in jobs_to_generate
        ]
        results_by_future = {}
        done = progress_done
        for future in as_completed(futures):
            results_by_future[future] = future.result()
            done += 1
            _report_generation_progress(done, total, enabled=progress)
        return tuple(results_by_future[future] for future in futures)


def _run_sequence_jobs(
    jobs_to_generate: list[SequenceJob],
    *,
    workers: int,
    output_root: Path | str | None,
    cache_root: Path | str | None,
    force_recompute: bool,
    physics_backend,
    settings,
    image_format: str,
    jpeg_quality: int,
    write_anomaly_preview: bool,
    run_first_line_first: bool = True,
    progress: bool = False,
):
    """Generate sequences serially or with coarse per-sequence parallelism.

    When ``run_first_line_first`` is True (default) and more than one sequence
    will run under ``workers > 1``, the first job completes before the remaining
    jobs start in parallel. That warms shared cache artefacts (clean optical,
    mesh-derived caches) so later workers are more likely to hit than race.

    When ``progress=True``, print ``done/total`` after each completed sequence.
    """
    from gummybear.datasets.sequence_generation import generate_smoke_sequence

    if not jobs_to_generate:
        return ()

    common_kwargs = {
        "output_root": output_root,
        "cache_root": cache_root,
        "force_recompute": force_recompute,
        "settings": settings,
        "image_format": image_format,
        "jpeg_quality": jpeg_quality,
        "write_anomaly_preview": write_anomaly_preview,
        "max_workers": workers,
    }
    total = len(jobs_to_generate)

    if workers == 1 or len(jobs_to_generate) == 1:
        generated = []
        for index, job in enumerate(jobs_to_generate, start=1):
            generated.append(
                generate_smoke_sequence(
                    job,
                    backend=physics_backend,
                    **common_kwargs,
                )
            )
            _report_generation_progress(index, total, enabled=progress)
        return tuple(generated)

    batch_kwargs = {
        "workers": workers,
        "output_root": output_root,
        "cache_root": cache_root,
        "force_recompute": force_recompute,
        "physics_backend": physics_backend,
        "settings": settings,
        "image_format": image_format,
        "jpeg_quality": jpeg_quality,
        "write_anomaly_preview": write_anomaly_preview,
        "common_kwargs": common_kwargs,
        "progress": progress,
        "progress_total": total,
    }

    if run_first_line_first:
        first = generate_smoke_sequence(
            jobs_to_generate[0],
            backend=physics_backend,
            **common_kwargs,
        )
        _report_generation_progress(1, total, enabled=progress)
        rest = _run_parallel_sequence_batch(
            jobs_to_generate[1:],
            progress_done=1,
            **batch_kwargs,
        )
        return (first, *rest)

    return _run_parallel_sequence_batch(
        jobs_to_generate,
        progress_done=0,
        **batch_kwargs,
    )


def run_generation_workbook(
    path: Path | str,
    *,
    repo_root: Path | str | None = None,
    stl_root: Path | str | None = None,
    output_root: Path | str | None = None,
    cache_root: Path | str | None = None,
    limit: int | None = None,
    sequence_id: str | None = None,
    reconcile_outputs: bool = False,
    remove_stale: bool = False,
    dry_run: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    run_first_line_first: bool = True,
    physics_backend=None,
    runtime_settings=None,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    write_anomaly_preview: bool = True,
    force_recompute: bool = False,
    use_persistent_cache: bool = True,
    verbose: bool = False,
    progress: bool = False,
):
    """Load a workbook path and generate its sequences.

    Short path for the common notebook / script pattern:

    ```text
    load_generation_workbook → validate_generation_plan
      → build_execution_plan → run_generation_plan
    ```

    ``repo_root`` resolves relative ``stl_path`` entries. ``stl_root``, when
    set, is preferred for that lookup so CAD can live outside the data root.
    The workbook ``stl_path`` string remains the cache-key identity input.
    ``output_root`` overrides workbook ``output_root`` when provided (and also
    anchors the default scenario ``_cache`` directory unless ``cache_root`` is
    set).

    When ``remove_stale=True``, blocking on-disk sequence directories (stale /
    incomplete) are deleted before generation so the run can proceed.

    Set ``parallel=True`` to generate independent sequences concurrently.
    Views within a sequence stay serial. With ``parallel=True`` and
    ``max_workers=None``, worker count defaults to
    :func:`default_parallel_workers` (``cpu_count - 2``, at least 1).

    ``run_first_line_first=True`` (default) awaits the first sequence before
    starting the parallel pool so shared cache artefacts are more likely to
    be published once.

    ``verbose=False`` (default) stores a compact job summary on the returned
    :class:`~gummybear.datasets.sequence_generation.GenerationRunResult`
    (``cache`` / ``computed`` / orphans). Pass ``verbose=True`` for the wide
    diagnostic table used by existing milestone notebooks.

    ``progress=True`` prints ``done/total`` after each completed sequence job.
    """
    workbook = load_generation_workbook(path)
    plan = validate_generation_plan(
        workbook,
        repo_root=repo_root,
        stl_root=stl_root,
    )
    execution_plan = build_execution_plan(
        plan,
        limit=limit,
        sequence_id=sequence_id,
        cache_root=cache_root,
        output_root=output_root,
        reconcile_outputs=reconcile_outputs,
    )
    return run_generation_plan(
        execution_plan,
        output_root=output_root,
        limit=limit,
        sequence_id=sequence_id,
        remove_stale=remove_stale,
        dry_run=dry_run,
        parallel=parallel,
        max_workers=max_workers,
        run_first_line_first=run_first_line_first,
        physics_backend=physics_backend,
        runtime_settings=runtime_settings,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        write_anomaly_preview=write_anomaly_preview,
        force_recompute=force_recompute,
        use_persistent_cache=use_persistent_cache,
        verbose=verbose,
        progress=progress,
    )


def run_generation_plan(
    execution_plan: ExecutionPlan,
    *,
    output_root: Path | str | None = None,
    limit: int | None = None,
    sequence_id: str | None = None,
    remove_stale: bool = False,
    dry_run: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    run_first_line_first: bool = True,
    physics_backend=None,
    runtime_settings=None,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    write_anomaly_preview: bool = True,
    force_recompute: bool = False,
    use_persistent_cache: bool = True,
    verbose: bool = False,
    progress: bool = False,
):
    """Execute selected sequence jobs with persistent source caches.

    Default execution is serial (``parallel=False``). Optional coarse-grained
    parallelism runs **one worker per sequence**; camera views within a
    sequence remain serial. With ``parallel=True`` and ``max_workers=None``,
    workers default to :func:`default_parallel_workers`.

    When ``parallel=True``, ``run_first_line_first=True`` (default) completes
    the first missing sequence before launching the remaining jobs so shared
    source caches can be published once.

    Corruption generation is still refused. When ``remove_stale=True``,
    requested sequence directories that reconcile as stale or incomplete are
    deleted so generation can replace them. Orphans and disabled outputs are
    never removed. ``remove_stale`` cannot be combined with ``dry_run=True``.

    ``verbose`` is stored on the returned
    :class:`~gummybear.datasets.sequence_generation.GenerationRunResult` and
    controls how ``repr(result)`` / ``print(result)`` render.

    ``progress=True`` prints ``done/total`` after each completed sequence job.

    Notebook / protocol:
        M6 batch sequence generation entry point.
    """
    from gummybear.datasets.sequence_generation import (
        GenerationRunResult,
        SmokeRuntimeSettings,
    )
    from gummybear.datasets.output_plan import (
        OUTPUT_MISSING,
        OutputPlanError,
        build_output_delta_plan,
        remove_blocking_sequence_outputs,
    )

    workers = resolve_generation_workers(parallel=parallel, max_workers=max_workers)
    if image_format.strip().lower() not in {"jpg", "jpeg"}:
        raise GenerationPlanError("M6 Phase 2 dataset role images must use JPG.")
    if not 1 <= int(jpeg_quality) <= 100:
        raise GenerationPlanError("jpeg_quality must be between 1 and 100.")
    if force_recompute and not use_persistent_cache:
        raise GenerationPlanError("force_recompute requires use_persistent_cache=True.")
    if remove_stale and dry_run:
        raise GenerationPlanError(
            "remove_stale=True cannot be combined with dry_run=True "
            "(refusing to delete outputs during a dry run)."
        )

    selected_settings = (
        SmokeRuntimeSettings() if runtime_settings is None else runtime_settings
    )
    jobs = sorted(execution_plan.jobs, key=lambda item: item.sequence_id)
    if sequence_id is not None:
        jobs = [job for job in jobs if job.sequence_id == sequence_id]
        if not jobs:
            raise GenerationPlanError(
                f"No selected sequence_id={sequence_id!r} in execution plan."
            )
    if limit is not None:
        if limit < 0:
            raise GenerationPlanError(f"limit must be >= 0, got {limit}")
        jobs = jobs[:limit]

    for job in jobs:
        corruption_enabled = (
            job.corruption.enabled
            and job.corruption.corruption_kind.strip().lower() != "none"
        )
        if corruption_enabled:
            raise GenerationPlanError(
                "M6 Phase 2 does not generate corruptions; "
                f"sequence_id={job.sequence_id!r} requested "
                f"{job.corruption.corruption_kind!r}."
            )

    resolved_jobs = [
        resolve_job_output_identity(
            job,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            write_anomaly_preview=write_anomaly_preview,
            runtime_settings=asdict(selected_settings),
        )
        for job in jobs
    ]
    output_delta = build_output_delta_plan(
        resolved_jobs,
        output_root=output_root,
        disabled_sequence_ids=execution_plan.disabled_sequence_ids,
        scan_orphans=True,
    )
    if remove_stale and output_delta.blocking_items:
        remove_blocking_sequence_outputs(output_delta)
        output_delta = build_output_delta_plan(
            resolved_jobs,
            output_root=output_root,
            disabled_sequence_ids=execution_plan.disabled_sequence_ids,
            scan_orphans=True,
        )
    try:
        output_delta.require_safe_generation()
    except OutputPlanError as exc:
        raise GenerationPlanError(str(exc)) from exc

    missing_ids = {
        item.sequence_id
        for item in output_delta.requested
        if item.status == OUTPUT_MISSING
    }
    jobs_to_generate = [job for job in resolved_jobs if job.sequence_id in missing_ids]
    complete_ids = output_delta.complete_sequence_ids

    if dry_run:
        return GenerationRunResult(
            generated=(),
            skipped=tuple(item.sequence_id for item in output_delta.requested),
            dry_run=True,
            output_items=(
                output_delta.requested + output_delta.disabled + output_delta.orphaned
            ),
            verbose=verbose,
        )

    generated = _run_sequence_jobs(
        jobs_to_generate,
        workers=workers,
        output_root=output_root,
        cache_root=(execution_plan.cache_root if use_persistent_cache else None),
        force_recompute=force_recompute,
        physics_backend=physics_backend,
        settings=selected_settings,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        write_anomaly_preview=write_anomaly_preview,
        run_first_line_first=run_first_line_first,
        progress=progress,
    )
    return GenerationRunResult(
        generated=generated,
        skipped=complete_ids,
        output_items=(
            output_delta.requested + output_delta.disabled + output_delta.orphaned
        ),
        verbose=verbose,
    )
