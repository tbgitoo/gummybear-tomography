"""Per-sequence output reconciliation and safe delta planning.

Compares on-disk sequence directories against ``resolved_job_hash``, role files,
and cache provenance before generation overwrites or refuses stale outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from gummybear.datasets.generation_plan import (
    M6_5_MANIFEST_SCHEMA_VERSION,
    SequenceJob,
)
from gummybear.datasets.role_images import (
    jpg_relative_to_raw_tif,
    role_image_relative_to_raw_tif,
)
from gummybear.datasets.sequence_writer import frame_filename
from gummybear.paths import repo_relative_path

OUTPUT_MISSING = "output_missing"
OUTPUT_COMPLETE_CURRENT = "output_complete_current"
OUTPUT_INCOMPLETE = "output_incomplete"
OUTPUT_STALE_JOB_HASH = "output_stale_job_hash"
OUTPUT_STALE_SCHEMA = "output_stale_schema"
OUTPUT_STALE_CACHE_IDS = "output_stale_cache_ids"
OUTPUT_STALE_FRAME_MANIFEST = "output_stale_frame_manifest"
OUTPUT_ORPHANED_NOT_REQUESTED = "output_orphaned_not_requested"
OUTPUT_DISABLED_NOT_RUN = "output_disabled_not_run"

BLOCKING_OUTPUT_STATUSES = frozenset(
    {
        OUTPUT_INCOMPLETE,
        OUTPUT_STALE_JOB_HASH,
        OUTPUT_STALE_SCHEMA,
        OUTPUT_STALE_CACHE_IDS,
        OUTPUT_STALE_FRAME_MANIFEST,
    }
)


class OutputPlanError(ValueError):
    """Raised when existing outputs block safe generation or overwrite."""


@dataclass(frozen=True)
class OutputDeltaItem:
    """Classification of one sequence directory on disk.

    Attributes:
        sequence_id: Sequence identifier.
        output_path: Absolute path to the sequence directory (or intended path).
        status: One of the ``OUTPUT_*`` constants (e.g. ``output_missing``).
        reason: Machine-readable sub-reason for the status.
        details: Extra structured fields (expected hashes, missing files, etc.).
    """

    sequence_id: str
    output_path: str
    status: str
    reason: str
    details: dict[str, Any]

    @property
    def blocking(self) -> bool:
        """True when generation must not proceed without explicit stale removal."""
        return self.status in BLOCKING_OUTPUT_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON file-friendly dict with repo-relative ``output_path``."""
        return {
            "sequence_id": self.sequence_id,
            "output_path": repo_relative_path(self.output_path),
            "status": self.status,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class OutputDeltaPlan:
    """Reconciliation result for requested, disabled, and orphaned sequences.

    Attributes:
        requested: One item per enabled job after ``resolve_job_output_identity``.
        disabled: Workbook-disabled ids that may still have on-disk folders.
        orphaned: Existing sequence dirs under scenario roots not in the workbook.
    """

    requested: tuple[OutputDeltaItem, ...]
    disabled: tuple[OutputDeltaItem, ...] = ()
    orphaned: tuple[OutputDeltaItem, ...] = ()

    @property
    def blocking_items(self) -> tuple[OutputDeltaItem, ...]:
        """Requested items whose status forbids silent overwrite."""
        return tuple(item for item in self.requested if item.blocking)

    @property
    def missing_sequence_ids(self) -> tuple[str, ...]:
        """Sequence ids with no current output directory."""
        return tuple(
            item.sequence_id for item in self.requested if item.status == OUTPUT_MISSING
        )

    @property
    def complete_sequence_ids(self) -> tuple[str, ...]:
        """Sequence ids whose on-disk artifacts match the resolved job identity."""
        return tuple(
            item.sequence_id
            for item in self.requested
            if item.status == OUTPUT_COMPLETE_CURRENT
        )

    def require_safe_generation(self) -> None:
        """Raise :class:`OutputPlanError` when any requested item is blocking."""
        if not self.blocking_items:
            return
        descriptions = "; ".join(
            f"{item.sequence_id}: {item.status} ({item.reason})"
            for item in self.blocking_items
        )
        raise OutputPlanError(
            "Existing sequence outputs are stale or incomplete; refusing to "
            f"run physics or overwrite them: {descriptions}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON file-friendly plan summary."""
        return {
            "requested": [item.to_dict() for item in self.requested],
            "disabled": [item.to_dict() for item in self.disabled],
            "orphaned": [item.to_dict() for item in self.orphaned],
        }


def remove_blocking_sequence_outputs(
    delta: OutputDeltaPlan,
) -> tuple[Path, ...]:
    """Delete on-disk sequence directories for blocking *requested* items.

    Orphaned and disabled entries are never removed. Returns the absolute
    paths that were deleted.
    """
    import shutil

    removed: list[Path] = []
    for item in delta.blocking_items:
        path = Path(item.output_path)
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path.resolve())
    return tuple(removed)


def resolve_output_root(
    job: SequenceJob,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve scenario output root using the same policy as generation.

    Relative ``output_root`` values anchor to the repository inferred from the
    workbook path (``configs/`` parent when present).

    Args:
        job: Sequence job carrying ``output_root`` and ``workbook_path``.
        output_root: Optional override of the job's configured root.

    Returns:
        Absolute or cwd-relative path to the scenario directory.
    """
    configured = Path(job.output_root if output_root is None else output_root)
    if configured.is_absolute():
        return configured

    workbook_path = Path(job.workbook_path).resolve()
    if workbook_path.parent.name == "configs":
        return workbook_path.parent.parent / configured

    for parent in workbook_path.parents:
        if parent.name == "configs":
            return parent.parent / configured
        candidate = parent / configured
        if candidate.is_dir() or (parent / "configs").is_dir():
            return candidate
    return Path.cwd() / configured


def _expected_role_files(job: SequenceJob) -> dict[str, tuple[str, ...]]:
    count = len(job.camera.poses)
    index_width = max(4, len(str(count - 1)))
    roles: dict[str, list[str]] = {
        "clean": [],
        "particle": [],
        "observed": [],
        "clean_raw": [],
        "particle_raw": [],
        "observed_raw": [],
    }
    if job.write_anomaly_preview:
        roles["anomaly"] = []
        roles["anomaly_raw"] = []
    for pose in job.camera.poses:
        jpg_name = frame_filename(
            job.sequence_id,
            pose.frame_index,
            pose.angle_deg,
            extension="jpg",
            index_width=index_width,
        )
        for role in ("clean", "particle", "observed"):
            jpg_relative = f"{role}/{jpg_name}"
            roles[role].append(jpg_relative)
            roles[f"{role}_raw"].append(jpg_relative_to_raw_tif(jpg_relative))
        if job.write_anomaly_preview:
            anomaly_name = frame_filename(
                job.sequence_id,
                pose.frame_index,
                pose.angle_deg,
                extension="png",
                index_width=index_width,
            )
            anomaly_relative = f"anomaly/{anomaly_name}"
            roles["anomaly"].append(anomaly_relative)
            roles["anomaly_raw"].append(
                role_image_relative_to_raw_tif(anomaly_relative)
            )
    return {role: tuple(paths) for role, paths in roles.items()}


def _item(
    job: SequenceJob,
    sequence_dir: Path,
    status: str,
    reason: str,
    **details: Any,
) -> OutputDeltaItem:
    return OutputDeltaItem(
        sequence_id=job.sequence_id,
        output_path=str(sequence_dir),
        status=status,
        reason=reason,
        details=details,
    )


def reconcile_sequence_output(
    job: SequenceJob,
    *,
    output_root: Path | str | None = None,
) -> OutputDeltaItem:
    """Classify one requested sequence directory without mutating artifacts.

    Checks ``manifest.json``, ``resolved_job_hash``, role JPG/PNG/raw sidecars,
    cache ids, and per-frame filename alignment.

    Args:
        job: Job with ``resolved_job_hash`` and expected camera poses.
        output_root: Optional scenario root override.

    Returns:
        OutputDeltaItem describing missing, complete, stale, or incomplete state.
    """
    sequence_dir = resolve_output_root(job, output_root) / job.sequence_id
    if not sequence_dir.exists():
        return _item(
            job,
            sequence_dir,
            OUTPUT_MISSING,
            "sequence_directory_not_found",
        )
    if not sequence_dir.is_dir():
        return _item(
            job,
            sequence_dir,
            OUTPUT_INCOMPLETE,
            "sequence_path_is_not_directory",
        )

    manifest_path = sequence_dir / "manifest.json"
    if not manifest_path.is_file():
        return _item(job, sequence_dir, OUTPUT_INCOMPLETE, "manifest_not_found")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _item(
            job,
            sequence_dir,
            OUTPUT_INCOMPLETE,
            "manifest_unreadable",
            error=str(exc),
        )
    if not isinstance(manifest, dict):
        return _item(job, sequence_dir, OUTPUT_INCOMPLETE, "manifest_not_object")

    actual_schema = manifest.get("schema_version")
    if actual_schema != M6_5_MANIFEST_SCHEMA_VERSION:
        return _item(
            job,
            sequence_dir,
            OUTPUT_STALE_SCHEMA,
            "unsupported_manifest_schema",
            expected=M6_5_MANIFEST_SCHEMA_VERSION,
            actual=actual_schema,
        )

    actual_hash = manifest.get("resolved_job_hash")
    if actual_hash != job.resolved_job_hash:
        return _item(
            job,
            sequence_dir,
            OUTPUT_STALE_JOB_HASH,
            "resolved_job_hash_mismatch",
            expected=job.resolved_job_hash,
            actual=actual_hash,
        )

    expected_files = _expected_role_files(job)
    missing_files = sorted(
        relative
        for paths in expected_files.values()
        for relative in paths
        if not (sequence_dir / relative).is_file()
    )
    if missing_files:
        return _item(
            job,
            sequence_dir,
            OUTPUT_INCOMPLETE,
            "expected_role_files_missing",
            missing_files=missing_files,
        )

    caches = manifest.get("caches")
    expected_cache_ids = {
        "clean_optical_cache_id": job.clean_optical_cache_id,
        "particle_source_cache_id": job.particle_source_cache_id,
    }
    if not isinstance(caches, dict):
        return _item(
            job,
            sequence_dir,
            OUTPUT_STALE_CACHE_IDS,
            "cache_provenance_missing",
            expected=expected_cache_ids,
        )
    actual_cache_ids = {key: caches.get(key) for key in expected_cache_ids}
    if actual_cache_ids != expected_cache_ids:
        return _item(
            job,
            sequence_dir,
            OUTPUT_STALE_CACHE_IDS,
            "cache_provenance_mismatch",
            expected=expected_cache_ids,
            actual=actual_cache_ids,
        )

    expected_roles = {
        "clean": "clean",
        "particle": "particle",
        "observed": "observed",
    }
    if job.write_anomaly_preview:
        expected_roles["anomaly_preview"] = "anomaly"
    if manifest.get("roles") != expected_roles:
        return _item(
            job,
            sequence_dir,
            OUTPUT_STALE_FRAME_MANIFEST,
            "role_mapping_mismatch",
            expected=expected_roles,
            actual=manifest.get("roles"),
        )

    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != len(job.camera.poses):
        return _item(
            job,
            sequence_dir,
            OUTPUT_STALE_FRAME_MANIFEST,
            "frame_count_mismatch",
            expected=len(job.camera.poses),
            actual=len(frames) if isinstance(frames, list) else None,
        )

    for frame, pose in zip(frames, job.camera.poses, strict=True):
        if not isinstance(frame, dict):
            return _item(
                job,
                sequence_dir,
                OUTPUT_STALE_FRAME_MANIFEST,
                "frame_entry_not_object",
            )
        expected_filenames = {
            role: paths[pose.frame_index] for role, paths in expected_files.items()
        }
        actual_resolution = frame.get("resolution")
        angle_value = frame.get("angle_deg")
        try:
            actual_angle = (
                float(angle_value)
                if isinstance(angle_value, (int, float, str))
                else float("nan")
            )
        except (TypeError, ValueError):
            actual_angle = float("nan")
        if (
            frame.get("frame_index") != pose.frame_index
            or actual_angle != float(pose.angle_deg)
            or actual_resolution != [pose.resolution_y, pose.resolution_x]
            or frame.get("filenames") != expected_filenames
        ):
            return _item(
                job,
                sequence_dir,
                OUTPUT_STALE_FRAME_MANIFEST,
                "frame_manifest_mismatch",
                frame_index=pose.frame_index,
            )

    for role, paths in expected_files.items():
        if role.endswith("_raw"):
            directory = role[: -len("_raw")]
            extension = "*.raw.tif"
        elif role == "anomaly":
            directory = role
            extension = "*.png"
        else:
            directory = role
            extension = "*.jpg"
        actual_names = sorted(
            f"{directory}/{path.name}"
            for path in (sequence_dir / directory).glob(extension)
        )
        if actual_names != sorted(paths):
            return _item(
                job,
                sequence_dir,
                OUTPUT_STALE_FRAME_MANIFEST,
                "filename_order_or_membership_mismatch",
                role=role,
            )
        for relative in paths:
            try:
                with Image.open(sequence_dir / relative) as image:
                    if image.size != (job.camera.resolution_x, job.camera.resolution_y):
                        return _item(
                            job,
                            sequence_dir,
                            OUTPUT_STALE_FRAME_MANIFEST,
                            "image_dimensions_mismatch",
                            file=relative,
                            expected=[
                                job.camera.resolution_x,
                                job.camera.resolution_y,
                            ],
                            actual=list(image.size),
                        )
            except (OSError, UnidentifiedImageError) as exc:
                return _item(
                    job,
                    sequence_dir,
                    OUTPUT_INCOMPLETE,
                    "role_file_unreadable",
                    file=relative,
                    error=str(exc),
                )

    return _item(
        job,
        sequence_dir,
        OUTPUT_COMPLETE_CURRENT,
        "manifest_and_role_files_match",
    )


def build_output_delta_plan(
    jobs: Iterable[SequenceJob],
    *,
    output_root: Path | str | None = None,
    disabled_sequence_ids: Iterable[str] = (),
    scan_orphans: bool = True,
) -> OutputDeltaPlan:
    """Reconcile requested jobs and report disabled and orphaned outputs.

    Args:
        jobs: Resolved sequence jobs (must include ``resolved_job_hash``).
        output_root: Optional scenario root override for all jobs.
        disabled_sequence_ids: Workbook-disabled ids to report separately.
        scan_orphans: When True, list sequence folders not requested or disabled.

    Returns:
        OutputDeltaPlan with ``requested``, ``disabled``, and ``orphaned`` items.
    """
    selected = tuple(sorted(jobs, key=lambda job: job.sequence_id))
    requested = tuple(
        reconcile_sequence_output(job, output_root=output_root) for job in selected
    )
    roots = {resolve_output_root(job, output_root) for job in selected}
    if not roots and output_root is not None:
        roots.add(Path(output_root))

    disabled: list[OutputDeltaItem] = []
    disabled_ids = tuple(sorted(set(disabled_sequence_ids)))
    for sequence_id in disabled_ids:
        root = next(iter(roots), Path(output_root or "data/generated"))
        disabled.append(
            OutputDeltaItem(
                sequence_id=sequence_id,
                output_path=str(root / sequence_id),
                status=OUTPUT_DISABLED_NOT_RUN,
                reason="workbook_row_disabled",
                details={},
            )
        )

    orphaned: list[OutputDeltaItem] = []
    if scan_orphans:
        requested_ids = {job.sequence_id for job in selected}
        disabled_id_set = set(disabled_ids)
        for root in sorted(roots):
            if not root.is_dir():
                continue
            for path in sorted(root.iterdir()):
                if (
                    not path.is_dir()
                    or path.name.startswith(".")
                    or path.name.startswith("_")
                    or path.name in requested_ids
                    or path.name in disabled_id_set
                ):
                    continue
                orphaned.append(
                    OutputDeltaItem(
                        sequence_id=path.name,
                        output_path=str(path),
                        status=OUTPUT_ORPHANED_NOT_REQUESTED,
                        reason="existing_output_not_requested",
                        details={},
                    )
                )

    return OutputDeltaPlan(
        requested=requested,
        disabled=tuple(disabled),
        orphaned=tuple(orphaned),
    )
