"""Phase 1 tests for M6 generation planning and dry-run grouping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gummybear.datasets.generation_plan import (
    GenerationPlanError,
    build_execution_plan,
    summarize_execution_plan,
    validate_generation_plan,
)
from gummybear.datasets.generation_workbook import (
    load_generation_workbook,
    write_example_generation_workbook,
    write_matrix_generation_workbook,
)

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repository root from {here}")


REPO_ROOT = _repo_root()
EXAMPLE_WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_generation_plan.xlsx"
MATRIX_WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_matrix_plan.xlsx"


@pytest.fixture(scope="module")
def example_workbook_path() -> Path:
    if not EXAMPLE_WORKBOOK.is_file():
        write_example_generation_workbook(EXAMPLE_WORKBOOK)
    return EXAMPLE_WORKBOOK


def test_example_dry_run_smoke_plan(
    example_workbook_path: Path,
    tmp_path: Path,
):
    workbook = load_generation_workbook(example_workbook_path)
    plan = validate_generation_plan(workbook, repo_root=REPO_ROOT)
    execution = build_execution_plan(plan, cache_root=tmp_path / "cache")
    summary = summarize_execution_plan(
        execution,
        disabled_sequence_count=len(plan.disabled_sequence_ids),
    )

    assert summary.enabled_sequence_count == 1
    assert summary.sequence_count == 1
    assert summary.frame_count == 6
    assert summary.clean_group_count == 1
    assert summary.particle_group_count == 1
    assert summary.diffusion_group_count == 1
    assert summary.resolutions == ((128, 128),)
    assert summary.plans_diffusion_operator_cache is False
    assert execution.plans_operator_cache is False
    assert summary.expected_clean_cache_misses == 1
    assert summary.expected_particle_cache_misses == 1

    job = plan.jobs[0]
    assert job.sequence_id == "bear_m6_smoke_001"
    assert job.camera.num_views == 6
    assert job.camera.resolution_x == 128
    assert job.camera.resolution_y == 128
    assert len(job.camera.poses) == 6
    assert [pose.frame_index for pose in job.camera.poses] == list(range(6))
    assert job.camera.poses[0].angle_deg == 0.0
    assert job.camera.poses[-1].angle_deg == 300.0
    assert job.clean_optical_cache_id
    assert job.particle_source_cache_id
    assert len(job.resolved_job_hash) == 64
    assert "extrapolation_length" in job.diffusion_provenance
    assert "operator_cache" not in job.diffusion_provenance
    assert job.diffusion_provenance["mu_a"] == job.optical.mu_a
    assert job.diffusion_provenance["mu_a_source"] == "optical_setup"
    material = job.optical.as_optical_material(g=job.diffusion.g)
    assert job.diffusion_provenance["D"] == material.diffusion_coefficient
    assert job.diffusion_provenance["D"] == job.optical.diffusion_coefficient(
        g=job.diffusion.g
    )
    assert (
        job.diffusion_provenance["D_source"] == "optical_material_diffusion_coefficient"
    )
    assert job.diffusion_provenance["g"] == job.diffusion.g
    assert job.diffusion_provenance["g_source"] == "diffusion_setup"
    assert "mu_a" not in job.diffusion.__dataclass_fields__
    assert "D" not in job.diffusion.__dataclass_fields__


def test_broken_cross_reference_fails(
    tmp_path: Path,
    example_workbook_path: Path,
):
    frames = pd.read_excel(
        example_workbook_path,
        sheet_name=None,
        engine="openpyxl",
    )
    frames["sequences"].loc[0, "optical_setup_id"] = "missing_optical"
    broken = tmp_path / "broken_xref.xlsx"
    with pd.ExcelWriter(broken, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    workbook = load_generation_workbook(broken)
    with pytest.raises(GenerationPlanError, match="missing_optical"):
        validate_generation_plan(workbook, repo_root=REPO_ROOT)


def test_duplicate_enabled_sequence_id_fails(
    tmp_path: Path,
    example_workbook_path: Path,
):
    frames = pd.read_excel(
        example_workbook_path,
        sheet_name=None,
        engine="openpyxl",
    )
    duplicate = frames["sequences"].copy()
    frames["sequences"] = pd.concat(
        [frames["sequences"], duplicate],
        ignore_index=True,
    )
    broken = tmp_path / "duplicate_sequence.xlsx"
    with pd.ExcelWriter(broken, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    workbook = load_generation_workbook(broken)
    with pytest.raises(GenerationPlanError, match="Duplicate enabled sequence_id"):
        validate_generation_plan(workbook, repo_root=REPO_ROOT)


def test_camera_only_change_does_not_change_clean_cache_id(
    tmp_path: Path,
    example_workbook_path: Path,
):
    frames = pd.read_excel(
        example_workbook_path,
        sheet_name=None,
        engine="openpyxl",
    )
    base_workbook = load_generation_workbook(example_workbook_path)
    base_plan = validate_generation_plan(base_workbook, repo_root=REPO_ROOT)
    base_clean = base_plan.jobs[0].clean_optical_cache_id
    base_particle = base_plan.jobs[0].particle_source_cache_id

    frames["camera_schedules"].loc[0, "resolution_x"] = 224
    frames["camera_schedules"].loc[0, "resolution_y"] = 224
    frames["camera_schedules"].loc[0, "num_views"] = 4
    frames["camera_schedules"].loc[0, "angle_stop_deg"] = 270.0
    changed = tmp_path / "camera_changed.xlsx"
    with pd.ExcelWriter(changed, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    changed_workbook = load_generation_workbook(changed)
    changed_plan = validate_generation_plan(changed_workbook, repo_root=REPO_ROOT)
    assert changed_plan.jobs[0].clean_optical_cache_id == base_clean
    assert changed_plan.jobs[0].particle_source_cache_id == base_particle
    assert changed_plan.jobs[0].camera.resolution_x == 224
    assert changed_plan.jobs[0].camera.num_views == 4
    assert changed_plan.jobs[0].resolved_job_hash != base_plan.jobs[0].resolved_job_hash


def test_robin_change_does_not_change_source_cache_ids(
    tmp_path: Path,
    example_workbook_path: Path,
):
    frames = pd.read_excel(
        example_workbook_path,
        sheet_name=None,
        engine="openpyxl",
    )
    base_workbook = load_generation_workbook(example_workbook_path)
    base_plan = validate_generation_plan(base_workbook, repo_root=REPO_ROOT)
    base_clean = base_plan.jobs[0].clean_optical_cache_id
    base_particle = base_plan.jobs[0].particle_source_cache_id

    frames["diffusion_setups"].loc[0, "extrapolation_length"] = 12.0
    frames["diffusion_setups"].loc[0, "alpha_direct"] = 1.0
    changed = tmp_path / "robin_changed.xlsx"
    with pd.ExcelWriter(changed, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    changed_workbook = load_generation_workbook(changed)
    changed_plan = validate_generation_plan(changed_workbook, repo_root=REPO_ROOT)
    assert changed_plan.jobs[0].clean_optical_cache_id == base_clean
    assert changed_plan.jobs[0].particle_source_cache_id == base_particle
    assert changed_plan.jobs[0].diffusion.extrapolation_length == 12.0
    assert changed_plan.jobs[0].diffusion.alpha_direct == 1.0
    assert changed_plan.jobs[0].diffusion_provenance["alpha_direct"] == 1.0
    assert changed_plan.jobs[0].resolved_job_hash != base_plan.jobs[0].resolved_job_hash


def test_diffusion_g_changes_derived_d_without_clean_cache_invalidation(
    tmp_path: Path,
    example_workbook_path: Path,
):
    frames = pd.read_excel(
        example_workbook_path,
        sheet_name=None,
        engine="openpyxl",
    )
    base_workbook = load_generation_workbook(example_workbook_path)
    base_plan = validate_generation_plan(base_workbook, repo_root=REPO_ROOT)
    base_job = base_plan.jobs[0]
    base_clean = base_job.clean_optical_cache_id
    base_d = base_job.diffusion_provenance["D"]

    frames["diffusion_setups"]["g"] = frames["diffusion_setups"]["g"].astype(float)
    frames["diffusion_setups"].loc[0, "g"] = 0.5
    changed = tmp_path / "g_changed.xlsx"
    with pd.ExcelWriter(changed, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    changed_workbook = load_generation_workbook(changed)
    changed_plan = validate_generation_plan(changed_workbook, repo_root=REPO_ROOT)
    changed_job = changed_plan.jobs[0]
    assert changed_job.clean_optical_cache_id == base_clean
    assert changed_job.diffusion.g == 0.5
    assert changed_job.diffusion_provenance["D"] != base_d
    assert changed_job.diffusion_provenance["D"] == (
        changed_job.optical.diffusion_coefficient(g=0.5)
    )
    assert changed_job.diffusion_provenance["g_source"] == "diffusion_setup"


def test_limit_filters_deterministically(example_workbook_path: Path):
    workbook = load_generation_workbook(example_workbook_path)
    plan = validate_generation_plan(workbook, repo_root=REPO_ROOT)
    execution = build_execution_plan(plan, limit=0)
    assert execution.jobs == ()
    summary = summarize_execution_plan(execution)
    assert summary.frame_count == 0


def test_resolved_job_hash_ignores_workbook_provenance_changes(
    tmp_path: Path,
    example_workbook_path: Path,
):
    frames = pd.read_excel(
        example_workbook_path,
        sheet_name=None,
        engine="openpyxl",
    )
    base = validate_generation_plan(
        load_generation_workbook(example_workbook_path),
        repo_root=REPO_ROOT,
    ).jobs[0]
    frames["sequences"].loc[0, "notes"] = "unrelated provenance-only edit"
    changed_path = tmp_path / "renamed_workbook.xlsx"
    with pd.ExcelWriter(changed_path, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    changed = validate_generation_plan(
        load_generation_workbook(changed_path),
        repo_root=REPO_ROOT,
    ).jobs[0]

    assert changed.workbook_sha256 != base.workbook_sha256
    assert changed.workbook_path != base.workbook_path
    assert changed.resolved_job_hash == base.resolved_job_hash


def test_resolved_job_hash_ignores_split_and_sequence_seed(
    example_workbook_path: Path,
):
    base = validate_generation_plan(
        load_generation_workbook(example_workbook_path),
        repo_root=REPO_ROOT,
    ).jobs[0]
    from dataclasses import replace

    from gummybear.datasets.generation_plan import resolve_job_output_identity

    other_split = "validation" if base.split != "validation" else "test"
    changed = resolve_job_output_identity(
        replace(base, split=other_split, seed=int(base.seed) + 99)
    )
    assert "split" not in changed.resolved_job_payload.get("sequence", {})
    assert "seed" not in changed.resolved_job_payload.get("sequence", {})
    assert changed.resolved_job_hash == base.resolved_job_hash


def test_matrix_workbook_has_shared_sources_and_prepared_delta_row():
    if not MATRIX_WORKBOOK.is_file():
        write_matrix_generation_workbook(MATRIX_WORKBOOK)
    workbook = load_generation_workbook(MATRIX_WORKBOOK)
    plan = validate_generation_plan(workbook, repo_root=REPO_ROOT)
    execution = build_execution_plan(plan)

    assert [job.sequence_id for job in plan.jobs] == [
        "bear_m6_matrix_001",
        "bear_m6_matrix_002",
        "bear_m6_matrix_003",
    ]
    assert plan.disabled_sequence_ids == ("bear_m6_matrix_004",)
    assert len({job.clean_optical_cache_id for job in plan.jobs}) == 1
    assert len({job.particle_source_cache_id for job in plan.jobs}) == 1
    assert len(execution.clean_groups) == 1
    assert (
        sum(
            len(group.camera_tasks)
            for particle in execution.clean_groups[0].particle_groups
            for group in particle.diffusion_groups
        )
        == 30
    )
