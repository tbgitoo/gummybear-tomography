"""M6.5 output-idempotency and delta-planning tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gummybear.datasets.generation_plan import (
    GenerationPlanError,
    build_execution_plan,
    run_generation_plan,
    validate_generation_plan,
)
from gummybear.datasets.generation_workbook import load_generation_workbook
from gummybear.datasets.output_plan import (
    OUTPUT_COMPLETE_CURRENT,
    OUTPUT_INCOMPLETE,
    OUTPUT_ORPHANED_NOT_REQUESTED,
    OUTPUT_STALE_CACHE_IDS,
    OUTPUT_STALE_FRAME_MANIFEST,
    OUTPUT_STALE_JOB_HASH,
    OUTPUT_STALE_SCHEMA,
    build_output_delta_plan,
    reconcile_sequence_output,
)
from gummybear.datasets.sequence_generation import CapturedFrame

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repository root from {here}")


REPO_ROOT = _repo_root()
WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_generation_plan.xlsx"
MATRIX_WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_matrix_plan.xlsx"


class CountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    def prepare_clean(self, job, settings):
        self.calls += 1
        return {}

    def prepare_particle(self, job, clean_state, settings):
        self.calls += 1
        return {}

    def solve_fields(self, job, clean_state, particle_state, settings):
        self.calls += 1
        return {}

    def capture_frame(
        self,
        job,
        pose,
        clean_state,
        particle_state,
        field_state,
        settings,
    ):
        self.calls += 1
        shape = (pose.resolution_y, pose.resolution_x)
        clean = np.ones(shape)
        return CapturedFrame(
            clean=clean,
            particle=clean + 0.1,
            metadata={
                "frame_index": pose.frame_index,
                "angle_deg": pose.angle_deg,
                "axis": list(pose.axis),
                "camera_position": [0.0, -80.0, 2.5],
                "look_at": [0.0, 0.0, 2.5],
                "up": [0.0, 0.0, 1.0],
                "camera_kind": pose.camera_kind,
                "resolution": list(shape),
                "fov_deg": settings.camera_fov_deg,
            },
        )

    def diagnostics(self, clean_state, particle_state, field_state):
        return {"test": True}


@pytest.fixture
def job_and_execution():
    workbook = load_generation_workbook(WORKBOOK)
    plan = validate_generation_plan(workbook, repo_root=REPO_ROOT)
    return plan.jobs[0], build_execution_plan(plan, limit=1)


def _generate(execution, output_root: Path):
    backend = CountingBackend()
    result = run_generation_plan(
        execution,
        output_root=output_root,
        physics_backend=backend,
        use_persistent_cache=False,
        verbose=True,
    )
    return Path(result.generated[0].output_path), backend


def _edit_manifest(sequence_dir: Path, edit) -> None:
    path = sequence_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    edit(manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_missing_and_complete_outputs_are_distinct(job_and_execution, tmp_path: Path):
    job, execution = job_and_execution
    assert reconcile_sequence_output(job, output_root=tmp_path).status == (
        "output_missing"
    )
    sequence_dir, _ = _generate(execution, tmp_path)
    assert sequence_dir.is_dir()
    assert (
        reconcile_sequence_output(job, output_root=tmp_path).status
        == OUTPUT_COMPLETE_CURRENT
    )


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        (
            lambda manifest: manifest.__setitem__(
                "schema_version", "unsupported-schema"
            ),
            OUTPUT_STALE_SCHEMA,
        ),
        (
            lambda manifest: manifest.__setitem__("resolved_job_hash", "0" * 64),
            OUTPUT_STALE_JOB_HASH,
        ),
        (
            lambda manifest: manifest["caches"].__setitem__(
                "clean_optical_cache_id", "0" * 64
            ),
            OUTPUT_STALE_CACHE_IDS,
        ),
        (
            lambda manifest: manifest["frames"][0].__setitem__("resolution", [1, 1]),
            OUTPUT_STALE_FRAME_MANIFEST,
        ),
    ],
)
def test_manifest_mismatches_are_classified(
    job_and_execution,
    tmp_path: Path,
    mutation,
    expected_status: str,
):
    job, execution = job_and_execution
    sequence_dir, _ = _generate(execution, tmp_path)
    _edit_manifest(sequence_dir, mutation)
    assert (
        reconcile_sequence_output(job, output_root=tmp_path).status == expected_status
    )


def test_missing_role_file_is_incomplete(job_and_execution, tmp_path: Path):
    job, execution = job_and_execution
    sequence_dir, _ = _generate(execution, tmp_path)
    next((sequence_dir / "clean").glob("*.jpg")).unlink()
    assert (
        reconcile_sequence_output(job, output_root=tmp_path).status == OUTPUT_INCOMPLETE
    )


def test_stale_output_fails_before_physics(job_and_execution, tmp_path: Path):
    _, execution = job_and_execution
    sequence_dir, _ = _generate(execution, tmp_path)
    _edit_manifest(
        sequence_dir,
        lambda manifest: manifest.__setitem__("resolved_job_hash", "0" * 64),
    )
    backend = CountingBackend()

    with pytest.raises(GenerationPlanError, match="output_stale_job_hash"):
        run_generation_plan(
            execution,
            output_root=tmp_path,
            physics_backend=backend,
            use_persistent_cache=False,
            verbose=True,
        )
    assert backend.calls == 0


def test_remove_stale_deletes_and_regenerates(job_and_execution, tmp_path: Path):
    _, execution = job_and_execution
    sequence_dir, _ = _generate(execution, tmp_path)
    marker = sequence_dir / "marker.txt"
    marker.write_text("stale", encoding="utf-8")
    _edit_manifest(
        sequence_dir,
        lambda manifest: manifest.__setitem__("resolved_job_hash", "0" * 64),
    )
    backend = CountingBackend()

    result = run_generation_plan(
        execution,
        output_root=tmp_path,
        physics_backend=backend,
        use_persistent_cache=False,
        remove_stale=True,
        verbose=True,
    )

    assert backend.calls >= 1
    assert len(result.generated) == 1
    assert not marker.exists()
    assert sequence_dir.is_dir()
    assert (sequence_dir / "manifest.json").is_file()


def test_remove_stale_rejected_with_dry_run(job_and_execution, tmp_path: Path):
    _, execution = job_and_execution
    with pytest.raises(GenerationPlanError, match="remove_stale"):
        run_generation_plan(
            execution,
            output_root=tmp_path,
            dry_run=True,
            remove_stale=True,
            use_persistent_cache=False,
            verbose=True,
        )


def test_disabled_and_orphaned_outputs_are_reported_not_deleted(
    job_and_execution,
    tmp_path: Path,
):
    job, _ = job_and_execution
    orphan = tmp_path / "old_sequence"
    orphan.mkdir()
    plan = build_output_delta_plan(
        [job],
        output_root=tmp_path,
        disabled_sequence_ids=["disabled_sequence"],
    )

    assert plan.disabled[0].status == "output_disabled_not_run"
    assert plan.orphaned[0].status == OUTPUT_ORPHANED_NOT_REQUESTED
    assert orphan.is_dir()


def test_complete_outputs_are_removed_from_source_cache_plan(
    job_and_execution,
    tmp_path: Path,
):
    _, execution = job_and_execution
    _generate(execution, tmp_path)
    workbook = load_generation_workbook(WORKBOOK)
    plan = validate_generation_plan(workbook, repo_root=REPO_ROOT)

    reconciled = build_execution_plan(
        plan,
        cache_root=tmp_path / "absent_cache",
        output_root=tmp_path,
        reconcile_outputs=True,
    )

    assert reconciled.clean_groups == ()
    requested = [
        item
        for item in reconciled.output_items
        if item.sequence_id == "bear_m6_smoke_001"
    ]
    assert requested[0].status == OUTPUT_COMPLETE_CURRENT


def test_stl_path_resolves_for_temporary_workbook_outside_configs(
    tmp_path: Path,
):
    from gummybear.datasets.sequence_generation import _resolve_job_path

    frames = pd.read_excel(MATRIX_WORKBOOK, sheet_name=None, engine="openpyxl")
    temp_dir = tmp_path / "generated" / "_m6_5_workbooks"
    temp_dir.mkdir(parents=True)
    temp_workbook = temp_dir / "matrix_row_added.xlsx"
    with pd.ExcelWriter(temp_workbook, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    job = validate_generation_plan(
        load_generation_workbook(temp_workbook),
        repo_root=REPO_ROOT,
    ).jobs[0]
    resolved = _resolve_job_path(job)
    assert resolved.is_file()
    assert resolved.name == "proto_bear.stl"


def test_stl_path_resolves_for_temporary_workbook_outside_configs(
    tmp_path: Path,
):
    from dataclasses import replace

    from gummybear.datasets.sequence_generation import _resolve_job_path

    job = validate_generation_plan(
        load_generation_workbook(MATRIX_WORKBOOK),
        repo_root=REPO_ROOT,
    ).jobs[0]
    # Mimic the notebook workbook location under data/generated/.
    nested = (
        REPO_ROOT
        / "data"
        / "generated"
        / "_m6_5_workbooks"
        / "matrix_row_added.xlsx"
    )
    nested.parent.mkdir(parents=True, exist_ok=True)
    job = replace(job, workbook_path=str(nested))
    resolved = _resolve_job_path(job)
    assert resolved.is_file()
    assert resolved.resolve() == (REPO_ROOT / "cad" / "proto_bear.stl").resolve()


def test_enabling_matrix_row_generates_only_the_new_sequence(tmp_path: Path):
    initial_workbook = load_generation_workbook(MATRIX_WORKBOOK)
    initial_plan = validate_generation_plan(initial_workbook, repo_root=REPO_ROOT)
    initial_execution = build_execution_plan(
        initial_plan, cache_root=tmp_path / "cache"
    )
    first_backend = CountingBackend()
    first = run_generation_plan(
        initial_execution,
        output_root=tmp_path / "output",
        physics_backend=first_backend,
        use_persistent_cache=False,
        verbose=True,
    )
    assert [item.sequence_id for item in first.generated] == [
        "bear_m6_matrix_001",
        "bear_m6_matrix_002",
        "bear_m6_matrix_003",
    ]

    frames = pd.read_excel(MATRIX_WORKBOOK, sheet_name=None, engine="openpyxl")
    frames["sequences"].loc[
        frames["sequences"]["sequence_id"] == "bear_m6_matrix_004",
        "enabled",
    ] = True
    changed_workbook_path = tmp_path / "matrix_row_added.xlsx"
    with pd.ExcelWriter(changed_workbook_path, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    changed_plan = validate_generation_plan(
        load_generation_workbook(changed_workbook_path),
        repo_root=REPO_ROOT,
    )
    assert {
        job.sequence_id: job.resolved_job_hash for job in initial_plan.jobs
    }.items() <= {
        job.sequence_id: job.resolved_job_hash for job in changed_plan.jobs
    }.items()

    delta_execution = build_execution_plan(
        changed_plan,
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "output",
        reconcile_outputs=True,
    )
    second_backend = CountingBackend()
    second = run_generation_plan(
        delta_execution,
        output_root=tmp_path / "output",
        physics_backend=second_backend,
        use_persistent_cache=False,
        verbose=True,
    )

    assert [item.sequence_id for item in second.generated] == ["bear_m6_matrix_004"]
    assert set(second.skipped) == {
        "bear_m6_matrix_001",
        "bear_m6_matrix_002",
        "bear_m6_matrix_003",
    }
    assert second_backend.calls == 3 + 6
