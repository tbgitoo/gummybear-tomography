"""Manifest construction for generated multi-view sequences."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from gummybear import __version__
from gummybear.datasets.generation_plan import (
    M6_5_GENERATOR_VERSION,
    M6_5_MANIFEST_SCHEMA_VERSION,
    SequenceJob,
)


SCHEMA_VERSION = M6_5_MANIFEST_SCHEMA_VERSION
GENERATOR_VERSION = M6_5_GENERATOR_VERSION
LEGACY_SCHEMA_VERSIONS = frozenset(
    {"1.2-m6-draft", "1.3-m6-draft", "1.4-m6-draft", "1.5-m6-draft"}
)


def _portable_path(path_value: str, *, repository_root: Path) -> str:
    """Return a POSIX path without embedding a machine-specific root."""
    path = Path(path_value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        # Preserve a useful identity without leaking an external local root.
        return path.name


def _setup_with_workbook_coordinates(
    setup: Any,
    *,
    workbook_name: str,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Serialize a setup with explicit workbook cell coordinates."""
    values = {key: value for key, value in asdict(setup).items() if key not in exclude}
    values["workbook_name"] = workbook_name
    values["workbook_sheet"] = values["source_sheet"]
    return values


def build_sequence_manifest(
    job: SequenceJob,
    *,
    frame_metadata: Sequence[Mapping[str, Any]],
    runtime_settings: Mapping[str, Any],
    stage_seconds: Mapping[str, float],
    diagnostics: Mapping[str, Any],
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    anomaly_preview: bool = True,
    cache_events: Mapping[str, Any] | None = None,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Build the sequence ``manifest.json`` payload before role files are written.

    Role contract: ``clean`` (no particle), ``particle`` and ``observed`` (with
    particle; observed equals particle without corruption), optional ``anomaly``
    preview as ``particle_minus_clean`` in linear space before JPEG.

    Args:
        job: Resolved sequence job with cache ids and setups.
        frame_metadata: Per-frame pose/resolution entries (filenames added later).
        runtime_settings: Recorded runtime knobs (not cache keys).
        stage_seconds: Named timing breakdown from generation.
        diagnostics: Forward-model diagnostics (segment counts, residuals, etc.).
        image_format, jpeg_quality: Display encoding recorded in ``representation``.
        anomaly_preview: Whether anomaly preview roles are requested.
        cache_events: Optional map of cache kind → :class:`CacheEvent`.
        max_workers: Parallelism level recorded for audit (views stay serial).

    Returns:
        Manifest dict suitable for JSON serialization and output reconciliation.
    """
    workbook_path = Path(job.workbook_path)
    workbook_name = workbook_path.name
    repository_root = Path.cwd()
    if workbook_path.is_absolute():
        for parent in workbook_path.parents:
            if parent.name == "configs":
                repository_root = parent.parent
                break
    roles = {
        "clean": "clean",
        "particle": "particle",
        "observed": "observed",
    }
    if anomaly_preview:
        roles["anomaly_preview"] = "anomaly"

    cache_details = {}
    for name, event in (cache_events or {}).items():
        cache_details[name] = {
            "cache_id": str(event.cache_id),
            "status": str(event.status),
            "reason": str(event.reason),
            "load_seconds": float(event.load_seconds),
            "write_seconds": float(event.write_seconds),
        }
    persistent_cache_used = any(
        event.status != "disabled" for event in (cache_events or {}).values()
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "sequence_id": job.sequence_id,
        "resolved_job_hash": job.resolved_job_hash,
        "resolved_job": dict(job.resolved_job_payload),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forward_model_tier": job.forward_model_tier,
        "representation": {
            "image_format": image_format,
            "jpeg_quality": int(jpeg_quality),
            "image_domain": "camera_intensity",
            "composition_domain": "linear_camera_intensity_before_jpeg",
            "anomaly_definition": "particle_minus_clean",
            "composition_policy": "pre_jpeg_numeric_arrays",
            "observed_definition": "particle_no_corruption",
            "pixel_orientation": {
                "camera_up": "image_top",
                "transform_from_camera_sample_grid": "flip_axis_0",
            },
            "anomaly_preview": {
                "authoritative": False,
                "format": "png" if anomaly_preview else None,
                "mapping": "signed_per_frame_zero_centered"
                if anomaly_preview
                else None,
            },
        },
        "roles": roles,
        "phantom": {
            "phantom_id": job.phantom_id,
            "stl_path": _portable_path(
                job.stl_path,
                repository_root=repository_root,
            ),
            "stl_sha256": job.stl_sha256,
        },
        "workbook": {
            "workbook_path": _portable_path(
                job.workbook_path,
                repository_root=repository_root,
            ),
            "sha256": job.workbook_sha256,
            "sequence_sheet": "sequences",
            "sequence_excel_row": job.source_excel_row,
        },
        "setups": {
            "optical": _setup_with_workbook_coordinates(
                job.optical,
                workbook_name=workbook_name,
            ),
            "particle": {
                **_setup_with_workbook_coordinates(
                    job.particle,
                    workbook_name=workbook_name,
                ),
                "compatibility": "first_of_particles",
            },
            "particles": {
                "particle_group_id": job.particle_group_id,
                "count": len(job.particles),
                "order": "workbook_row_order",
                "items": [
                    _setup_with_workbook_coordinates(
                        item,
                        workbook_name=workbook_name,
                    )
                    for item in job.particles
                ],
            },
            "diffusion": {
                **_setup_with_workbook_coordinates(
                    job.diffusion,
                    workbook_name=workbook_name,
                ),
                "effective": dict(job.diffusion_provenance),
            },
            "camera": _setup_with_workbook_coordinates(
                job.camera,
                workbook_name=workbook_name,
                exclude=frozenset({"poses"}),
            ),
            "corruption": _setup_with_workbook_coordinates(
                job.corruption,
                workbook_name=workbook_name,
            ),
        },
        "caches": {
            "clean_optical_cache_id": job.clean_optical_cache_id,
            "particle_source_cache_id": job.particle_source_cache_id,
            "persistent_cache_used": persistent_cache_used,
            "events": cache_details,
            "diffusion_operator_cache": None,
        },
        "generation": {
            "package_version": __version__,
            "max_workers": int(max_workers),
            "runtime_settings": dict(runtime_settings),
            "stage_seconds": {
                key: float(value) for key, value in stage_seconds.items()
            },
            "diagnostics": dict(diagnostics),
        },
        "frames": [dict(frame) for frame in frame_metadata],
        "validation": {
            "role_alignment": "checked_before_write",
            "anomaly_identity": "particle_minus_clean_pre_jpeg",
            "post_jpeg_identity_checked": False,
        },
    }
