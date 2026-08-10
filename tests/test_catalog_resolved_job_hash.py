"""Catalog field_status must require matching resolved_job_hash."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from gummybear.datasets.generation_plan import (
    M6_5_MANIFEST_SCHEMA_VERSION,
    resolve_job_output_identity,
    validate_generation_plan,
)
from gummybear.datasets.generation_workbook import (
    load_generation_workbook,
    write_example_generation_workbook,
)
from tomography_ml.gummybear_data_catalog import build_catalog_rows
from tomography_ml.gummybear_data_catalog.catalog import build_catalog_row
from tomography_ml.gummybear_data_catalog.gummybear_adapter import (
    FIELD_STATUS_COMPLETE,
    FIELD_STATUS_STALE_JOB_HASH,
    reconcile_catalog_jobs_with_manifest,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repository root from {here}")


REPO_ROOT = _repo_root()
EXAMPLE_WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_test.xlsx"


@pytest.fixture
def planned_job(tmp_path: Path):
    if not EXAMPLE_WORKBOOK.is_file():
        write_example_generation_workbook(EXAMPLE_WORKBOOK)
    plan = validate_generation_plan(
        load_generation_workbook(EXAMPLE_WORKBOOK),
        repo_root=REPO_ROOT,
    )
    job = plan.jobs[0]
    output_root = tmp_path / "out"
    sequence_dir = output_root / job.sequence_id
    sequence_dir.mkdir(parents=True)
    return replace(job, output_root=str(output_root)), sequence_dir


def _write_manifest(
    sequence_dir: Path,
    *,
    sequence_id: str,
    schema: str,
    job_hash: object,
    split: str | None = None,
    seed: int | None = None,
) -> None:
    payload: dict = {
        "sequence_id": sequence_id,
        "schema_version": schema,
        "resolved_job_hash": job_hash,
        "roles": {},
        "representation": {"image_domain": "camera_intensity"},
    }
    # Legacy fields: optional; catalog must tolerate when present.
    if split is not None:
        payload["split"] = split
    if seed is not None:
        payload["seed"] = int(seed)
    (sequence_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def test_build_catalog_row_complete_when_hash_matches(planned_job) -> None:
    job, sequence_dir = planned_job
    _write_manifest(
        sequence_dir,
        sequence_id=job.sequence_id,
        schema=M6_5_MANIFEST_SCHEMA_VERSION,
        job_hash=job.resolved_job_hash,
    )
    row = build_catalog_row(sample_id=0, job=job)
    assert row.field_status == FIELD_STATUS_COMPLETE
    assert row.resolved_job_hash == job.resolved_job_hash


def test_build_catalog_row_complete_with_legacy_manifest_split_seed(planned_job) -> None:
    job, sequence_dir = planned_job
    _write_manifest(
        sequence_dir,
        sequence_id=job.sequence_id,
        schema=M6_5_MANIFEST_SCHEMA_VERSION,
        job_hash=job.resolved_job_hash,
        split="train",
        seed=999,
    )
    row = build_catalog_row(sample_id=0, job=job)
    assert row.field_status == FIELD_STATUS_COMPLETE
    # Catalog split still comes from the workbook job, not the legacy manifest field.
    assert row.split == job.split


def test_build_catalog_row_stale_when_hash_mismatches(planned_job) -> None:
    job, sequence_dir = planned_job
    _write_manifest(
        sequence_dir,
        sequence_id=job.sequence_id,
        schema=M6_5_MANIFEST_SCHEMA_VERSION,
        job_hash="0" * 64,
    )
    row = build_catalog_row(sample_id=0, job=job)
    assert row.field_status == FIELD_STATUS_STALE_JOB_HASH
    assert row.resolved_job_hash == job.resolved_job_hash


def test_build_catalog_row_stale_when_hash_missing(planned_job) -> None:
    job, sequence_dir = planned_job
    _write_manifest(
        sequence_dir,
        sequence_id=job.sequence_id,
        schema=M6_5_MANIFEST_SCHEMA_VERSION,
        job_hash="",
    )
    row = build_catalog_row(sample_id=0, job=job)
    assert row.field_status == FIELD_STATUS_STALE_JOB_HASH


def test_build_catalog_rows_and_reconcile_agree_on_stale(planned_job) -> None:
    job, sequence_dir = planned_job
    _write_manifest(
        sequence_dir,
        sequence_id=job.sequence_id,
        schema=M6_5_MANIFEST_SCHEMA_VERSION,
        job_hash="deadbeef",
    )
    rows = build_catalog_rows([job])
    assert len(rows) == 1
    assert rows[0].field_status == FIELD_STATUS_STALE_JOB_HASH

    reconcile = reconcile_catalog_jobs_with_manifest([job])
    assert reconcile.loc[0, "field_status"] == FIELD_STATUS_STALE_JOB_HASH


def test_split_change_keeps_hash_and_catalog_complete(planned_job) -> None:
    job, sequence_dir = planned_job
    _write_manifest(
        sequence_dir,
        sequence_id=job.sequence_id,
        schema=M6_5_MANIFEST_SCHEMA_VERSION,
        job_hash=job.resolved_job_hash,
    )
    other_split = "validation" if job.split != "validation" else "test"
    drifted = resolve_job_output_identity(
        replace(job, split=other_split, seed=int(job.seed) + 1)
    )
    assert drifted.resolved_job_hash == job.resolved_job_hash
    row = build_catalog_row(sample_id=0, job=drifted)
    assert row.field_status == FIELD_STATUS_COMPLETE
    assert row.split == other_split
