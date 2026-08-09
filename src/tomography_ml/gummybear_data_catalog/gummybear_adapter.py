"""Workbook-to-catalog adapter for generation plans and schedule identity.

Loads and validates Excel workbooks into ``SequenceJob`` lists, summarizes
jobs as DataFrames, and checks camera-schedule consistency across sequences.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from gummybear.datasets.generation_plan import (
    GenerationPlan,
    SequenceJob,
    validate_generation_plan,
)
from gummybear.datasets.generation_workbook import load_generation_workbook
from gummybear.datasets.output_plan import resolve_output_root

import pandas as pd

FIELD_STATUS_COMPLETE = "complete"
FIELD_STATUS_DIRECTORY_MISSING = "directory_missing"
FIELD_STATUS_MANIFEST_MISSING = "manifest_missing"
FIELD_STATUS_MANIFEST_INVALID = "manifest_invalid"
FIELD_STATUS_INCOMPLETE_CATALOG = "incomplete_catalog"
FIELD_STATUS_STALE_JOB_HASH = "stale_job_hash"

SCHEDULE_STATUS_CONSISTENT = "consistent"
SCHEDULE_STATUS_INCONSISTENT = "inconsistent"

ALLOWED_FIELD_STATUSES = frozenset(
    {
        FIELD_STATUS_COMPLETE,
        FIELD_STATUS_DIRECTORY_MISSING,
        FIELD_STATUS_MANIFEST_MISSING,
        FIELD_STATUS_MANIFEST_INVALID,
        FIELD_STATUS_INCOMPLETE_CATALOG,
        FIELD_STATUS_STALE_JOB_HASH,
    }
)

ALLOWED_SCHEDULE_STATUSES = frozenset(
    {
        SCHEDULE_STATUS_CONSISTENT,
        SCHEDULE_STATUS_INCONSISTENT,
    }
)


def manifest_resolved_job_hash_matches(
    job: SequenceJob,
    payload: Mapping[str, Any],
) -> bool:
    """Return True when the manifest ``resolved_job_hash`` equals the job hash.

    Comparison is strict. Train/validation/test ``split`` and workbook sequence
    ``seed`` are not part of generation identity, so they are not consulted.
    Legacy manifests may still contain those fields; they are ignored here.
    """
    manifest_hash = payload.get("resolved_job_hash")
    return (
        isinstance(manifest_hash, str)
        and bool(manifest_hash)
        and manifest_hash == job.resolved_job_hash
    )


REQUIRED_SCHEDULE_IDENTITY_COLUMNS = (
    "sequence_id",
    "camera_schedule_id",
    "frame_count",
    "resolution_x",
    "resolution_y",
    "first_angle_deg",
    "last_angle_deg",
    "angles_deg",
    "angles_hash",
    "schedule_status",
)


def _as_path(path: str | Path | Traversable) -> Path:
    """Normalize path-like / importlib resource paths to ``pathlib.Path``."""
    if isinstance(path, Path):
        return path
    if isinstance(path, str):
        return Path(path)
    # Traversable / MultiplexedPath from importlib.resources
    return Path(str(path))


def ordered_angles_deg(job: SequenceJob) -> tuple[float, ...]:
    """Return acquisition-order camera angles from ``job.camera.poses``.

    Angles follow ``frame_<index>`` ordering in generated sequences; they are
    never sorted. Shared by schedule-identity checks and catalog ``angles_deg``.

    Args:
        job: One validated workbook ``SequenceJob``.

    Returns:
        Tuple of ``angle_deg`` values in pose / frame order.
    """
    return tuple(float(pose.angle_deg) for pose in job.camera.poses)


def angles_hash(angles_deg: Sequence[float]) -> str:
    """Stable sha256 fingerprint of an ordered angle sequence.

    Order is significant: permuting angles yields a different hash. Values are
    serialized as ``repr(float(angle))`` joined by commas before hashing.

    Args:
        angles_deg: Camera angles in acquisition order (not sorted).

    Returns:
        Lowercase hex digest for schedule-consistency joins and catalog rows.
    """
    payload = ",".join(repr(float(angle)) for angle in angles_deg)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _schedule_identity(
    job: SequenceJob,
) -> tuple[str, int, int, int, tuple[float, ...]]:
    """Identity used to decide schedule consistency for one job."""
    return (
        str(job.camera.camera_schedule_id),
        int(job.camera.num_views),
        int(job.camera.resolution_x),
        int(job.camera.resolution_y),
        ordered_angles_deg(job),
    )


def load_catalog_plan(
    workbook_path: str | Path | Traversable,
    root_path: str | Path | Traversable | None = None,
    *,
    stl_root: str | Path | Traversable | None = None,
) -> GenerationPlan:
    """Validate a workbook into a ``GenerationPlan`` for catalog use.

    ``root_path`` is passed as ``repo_root`` (data / scenario root). Optional
    ``stl_root`` locates relative workbook ``stl_path`` entries when CAD lives
    outside that root. The stored relative ``stl_path`` string is unchanged and
    remains the cache-key identity input; only on-disk hashing lookup changes.
    """
    if root_path is None:
        root_path = Path.cwd()

    workbook = load_generation_workbook(_as_path(workbook_path))
    plan = validate_generation_plan(
        workbook,
        repo_root=_as_path(root_path),
        stl_root=None if stl_root is None else _as_path(stl_root),
    )

    return plan


def load_catalog_jobs(
    workbook_path: str | Path | Traversable,
    root_path: str | Path | Traversable | None = None,
    *,
    stl_root: str | Path | Traversable | None = None,
) -> list[SequenceJob]:
    """Return enabled ``SequenceJob`` rows for catalog construction.

    See :func:`load_catalog_plan` for ``root_path`` / ``stl_root`` semantics.
    """
    plan = load_catalog_plan(
        workbook_path=workbook_path,
        root_path=root_path,
        stl_root=stl_root,
    )
    return list(plan.jobs)


def catalog_jobs_to_dataframe(jobs: list[Any]) -> pd.DataFrame:
    """Summarize workbook ``SequenceJob`` rows as a flat pandas table.

    One row per input job with workbook metadata, resolved output paths, and
    generation identifiers. Does not read manifests or validate on-disk
    artifacts; use :func:`reconcile_catalog_jobs_with_manifest` for that.

    ``sample_id`` is the enumerate index in ``jobs``, not a persistent key
    across workbook reloads.

    Args:
        jobs: Job list from :func:`load_catalog_jobs` (or any ``SequenceJob``
            sequence).

    Returns:
        DataFrame with columns for split, paths, setup ids, cache keys, and
        workbook provenance.
    """
    rows: list[dict[str, Any]] = []

    for index, job in enumerate(jobs):
        rows.append(
            {
                "sample_id": index,
                "sequence_id": job.sequence_id,
                "split": job.split,
                "seed": job.seed,
                "output_root": job.output_root,
                "sequence_dir": str(resolve_output_root(job) / job.sequence_id),
                "phantom_id": job.phantom_id,
                "stl_path": job.stl_path,
                "forward_model_tier": job.forward_model_tier,
                "optical_setup_id": job.optical.optical_setup_id,
                "particle_setup_id": job.particle.particle_setup_id,
                "particle_group_id": job.particle_group_id,
                "n_particles": len(job.particles),
                "diffusion_setup_id": job.diffusion.diffusion_setup_id,
                "camera_schedule_id": job.camera.camera_schedule_id,
                "corruption_setup_id": job.corruption.corruption_setup_id,
                "num_views": job.camera.num_views,
                "resolution_x": job.camera.resolution_x,
                "resolution_y": job.camera.resolution_y,
                "particle_enabled": job.particle.enabled,
                "corruption_enabled": job.corruption.enabled,
                "enabled": job.enabled,
                "notes": job.notes,
                "source_excel_row": job.source_excel_row,
                "workbook_path": job.workbook_path,
                "workbook_sha256": job.workbook_sha256,
                "clean_optical_cache_id": job.clean_optical_cache_id,
                "particle_source_cache_id": job.particle_source_cache_id,
                "resolved_job_hash": job.resolved_job_hash,
                "selected_status": "workbook_enabled",
                "disabled_reported": False,
            }
        )

    return pd.DataFrame(rows)


def filter_schedule_consistent(
    catalog_jobs: Sequence[SequenceJob],
    *,
    camera_schedule_id: str,
) -> list[SequenceJob]:
    """Return jobs for one camera schedule with a single acquisition shape.

    Filters to ``camera_schedule_id`` and requires the remaining jobs to share
    the same ``V``, ordered angle list, and resolution. Does not mutate
    ``catalog_jobs``. Raises ``ValueError`` if the filtered subset is still
    mixed.
    """
    selected = [
        job
        for job in catalog_jobs
        if job.camera.camera_schedule_id == camera_schedule_id
    ]
    if not selected:
        return []

    identities = {_schedule_identity(job) for job in selected}
    if len(identities) != 1:
        raise ValueError(
            "Filtered jobs for "
            f"camera_schedule_id={camera_schedule_id!r} are not "
            "schedule-consistent: mixed V, ordered angles, or resolution."
        )
    return list(selected)


def schedule_identity_table(
    catalog_jobs: Sequence[SequenceJob],
) -> pd.DataFrame:
    """Return per-job schedule identity fields for validation and reporting.

    ``schedule_status`` is ``consistent`` when the provided collection shares a
    single schedule identity (or is empty); otherwise every row is marked
    ``inconsistent``. Angles are acquisition-ordered, never sorted.
    """
    jobs = list(catalog_jobs)
    identities = {_schedule_identity(job) for job in jobs}
    schedule_status = (
        SCHEDULE_STATUS_CONSISTENT
        if len(identities) <= 1
        else SCHEDULE_STATUS_INCONSISTENT
    )

    rows: list[dict[str, Any]] = []
    for job in jobs:
        angles = ordered_angles_deg(job)
        rows.append(
            {
                "sequence_id": job.sequence_id,
                "camera_schedule_id": job.camera.camera_schedule_id,
                "frame_count": int(job.camera.num_views),
                "resolution_x": int(job.camera.resolution_x),
                "resolution_y": int(job.camera.resolution_y),
                "first_angle_deg": angles[0] if angles else None,
                "last_angle_deg": angles[-1] if angles else None,
                "angles_deg": list(angles),
                "angles_hash": angles_hash(angles),
                "schedule_status": schedule_status,
            }
        )
    return pd.DataFrame(rows)


def reconcile_catalog_jobs_with_manifest(
    catalog_jobs: Sequence[SequenceJob],
    *,
    deep_validation: bool = False,
) -> pd.DataFrame:
    """Report on-disk manifest readiness for each workbook-defined catalog job.

    Catalog membership is unchanged: one output row is returned per input job.
    Ordinary artifact problems are reported via ``field_status`` and never raise.
    Image files are not opened unless ``deep_validation`` is explicitly enabled.
    """
    rows: list[dict[str, Any]] = []

    for sample_id, job in enumerate(catalog_jobs):
        sequence_dir = resolve_output_root(job) / job.sequence_id
        manifest_path = sequence_dir / "manifest.json"

        sequence_dir_exists = sequence_dir.is_dir()
        manifest_exists = sequence_dir_exists and manifest_path.is_file()
        manifest_sequence_id: str | None = None
        schema_version: str | None = None
        frame_count: int | None = None
        roles_present: list[str] | None = None

        if not sequence_dir_exists:
            field_status = FIELD_STATUS_DIRECTORY_MISSING
        elif not manifest_exists:
            field_status = FIELD_STATUS_MANIFEST_MISSING
        else:
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                field_status = FIELD_STATUS_MANIFEST_INVALID
            else:
                if not isinstance(payload, dict):
                    field_status = FIELD_STATUS_MANIFEST_INVALID
                else:
                    raw_sequence_id = payload.get("sequence_id")
                    if raw_sequence_id is not None:
                        manifest_sequence_id = str(raw_sequence_id)
                    raw_schema = payload.get("schema_version")
                    if raw_schema is not None:
                        schema_version = str(raw_schema)
                    frames = payload.get("frames", [])
                    if isinstance(frames, list):
                        frame_count = len(frames)
                    roles = payload.get("roles", {})
                    if isinstance(roles, dict):
                        roles_present = sorted(str(key) for key in roles.keys())

                    if (
                        manifest_sequence_id != job.sequence_id
                        or schema_version is None
                    ):
                        field_status = FIELD_STATUS_INCOMPLETE_CATALOG
                    elif manifest_resolved_job_hash_matches(job, payload):
                        field_status = FIELD_STATUS_COMPLETE
                    else:
                        field_status = FIELD_STATUS_STALE_JOB_HASH

        row: dict[str, Any] = {
            "sample_id": sample_id,
            "sequence_id": job.sequence_id,
            "sequence_dir_exists": sequence_dir_exists,
            "manifest_exists": manifest_exists,
            "manifest_sequence_id": manifest_sequence_id,
            "schema_version": schema_version,
            "frame_count": frame_count,
            "roles_present": roles_present,
            "field_status": field_status,
        }

        if deep_validation:
            deep_ok = False
            if field_status == FIELD_STATUS_COMPLETE:
                from gummybear_validation.milestone_06 import (
                    validate_generated_sequence,
                )

                deep_ok = bool(validate_generated_sequence(sequence_dir).ok)
            row["deep_validation_ok"] = deep_ok

        rows.append(row)

    return pd.DataFrame(rows)
