"""Multi-particle M6/M7 orchestration contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from gummybear.datasets.cache_keys import (
    clean_optical_cache_key,
    particle_source_cache_key,
    particle_source_cache_key_payload,
)
from gummybear.datasets.generation_plan import validate_generation_plan
from gummybear.datasets.generation_workbook import (
    attach_particle_group,
    example_workbook_frames,
    load_generation_workbook,
    write_example_generation_workbook,
    write_generation_workbook_frames,
)
from gummybear.datasets.manifest_writer import build_sequence_manifest
from gummybear.datasets.sequence_generation import DefaultSmokePhysicsBackend
from tomography_ml.gummybear_data_catalog import build_catalog_rows

REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_particle_item(**overrides):
    item = dict(
        particle_setup_id="p0",
        particle_kind="sphere",
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
        radius=3.0,
        mu_s_particle=0.8,
        mu_a_particle=0.2,
        refractive_index_particle=1.33,
    )
    item.update(overrides)
    return item


def test_particle_cache_payload_uses_ordered_particles_list():
    clean_id = clean_optical_cache_key(
        stl_sha256="abc123",
        illumination_kind="point",
        light_position_x=1.0,
        light_position_y=2.0,
        light_position_z=3.0,
        num_source_rays=512,
        mu_s=0.3,
        mu_a=0.1,
        refractive_index=1.33,
        source_deposition_method="exact_ray_tet_intervals",
    )
    payload = particle_source_cache_key_payload(
        clean_optical_cache_id=clean_id,
        particle_kind="sphere",
        center_x=1.0,
        center_y=2.0,
        center_z=3.0,
        radius=3.0,
        mu_s_particle=0.8,
        mu_a_particle=0.2,
        refractive_index_particle=1.33,
        placement_mode="fixed",
        particle_setup_id="solo",
    )
    assert "particle" not in payload
    assert payload["particles"][0]["particle_setup_id"] == "solo"
    assert len(payload["particles"]) == 1


def test_particle_cache_id_changes_when_either_particle_moves():
    clean_id = "clean" * 8
    base_particles = [
        _base_particle_item(particle_setup_id="a", center_x=1.0),
        _base_particle_item(particle_setup_id="b", center_x=2.0),
    ]
    key_a = particle_source_cache_key(
        clean_optical_cache_id=clean_id,
        particles=base_particles,
        placement_mode="fixed",
    )
    moved = [
        base_particles[0],
        _base_particle_item(particle_setup_id="b", center_x=2.5),
    ]
    key_b = particle_source_cache_key(
        clean_optical_cache_id=clean_id,
        particles=moved,
        placement_mode="fixed",
    )
    assert key_a != key_b

    reordered = [base_particles[1], base_particles[0]]
    key_reordered = particle_source_cache_key(
        clean_optical_cache_id=clean_id,
        particles=reordered,
        placement_mode="fixed",
    )
    assert key_a != key_reordered


def test_legacy_single_particle_workbook_still_parses(tmp_path: Path):
    path = write_example_generation_workbook(tmp_path / "smoke.xlsx")
    workbook = load_generation_workbook(path)
    plan = validate_generation_plan(workbook, repo_root=REPO_ROOT)
    assert len(plan.jobs) == 1
    job = plan.jobs[0]
    assert len(job.particles) == 1
    assert job.particle.particle_setup_id == job.particles[0].particle_setup_id
    assert job.particle_group_id == job.particle.particle_setup_id


def test_checked_in_multi_particle_workbook_parses():
    from gummybear.datasets.generation_workbook import (
        write_multi_particle_generation_workbook,
    )

    path = REPO_ROOT / "configs" / "m6" / "m6_multi_particle.xlsx"
    if not path.is_file():
        write_multi_particle_generation_workbook(path)
    plan = validate_generation_plan(
        load_generation_workbook(path),
        repo_root=REPO_ROOT,
    )
    job = plan.jobs[0]
    assert job.sequence_id == "bear_m6_multi_001"
    assert job.particle_group_id == "dryrun_two_sphere"
    assert len(job.particles) == 2
    assert job.output_root == "data/generated/m6_1"
    assert job.particles[0].center_x == pytest.approx(-5.0)
    assert job.particles[1].center_x == pytest.approx(5.0)


def test_write_multi_particle_generation_workbook(tmp_path: Path):
    from gummybear.datasets.generation_workbook import (
        write_multi_particle_generation_workbook,
    )

    path = write_multi_particle_generation_workbook(tmp_path / "multi.xlsx")
    plan = validate_generation_plan(
        load_generation_workbook(path),
        repo_root=REPO_ROOT,
    )
    assert len(plan.jobs[0].particles) == 2


def test_two_particle_group_parses_in_workbook_order(tmp_path: Path):
    frames = attach_particle_group(
        example_workbook_frames(),
        sequence_id="bear_m6_smoke_001",
        particle_group_id="group_two",
        centers=[
            (-5.0, 0.5, 2.5),
            (5.0, -0.5, 2.5),
        ],
    )
    path = write_generation_workbook_frames(tmp_path / "two.xlsx", frames)
    plan = validate_generation_plan(
        load_generation_workbook(path),
        repo_root=REPO_ROOT,
    )
    job = plan.jobs[0]
    assert len(job.particles) == 2
    assert job.particle_group_id == "group_two"
    assert job.particles[0].center_x == pytest.approx(-5.0)
    assert job.particles[1].center_x == pytest.approx(5.0)
    assert job.particle.particle_setup_id == job.particles[0].particle_setup_id


def test_manifest_and_catalog_expose_ordered_particles(tmp_path: Path):
    frames = attach_particle_group(
        example_workbook_frames(),
        sequence_id="bear_m6_smoke_001",
        particle_group_id="group_two",
        centers=[
            (-5.0, 0.5, 2.5),
            (5.0, -0.5, 2.5),
        ],
    )
    path = write_generation_workbook_frames(tmp_path / "two.xlsx", frames)
    plan = validate_generation_plan(
        load_generation_workbook(path),
        repo_root=REPO_ROOT,
    )
    job = plan.jobs[0]

    manifest = build_sequence_manifest(
        job,
        frame_metadata=[],
        runtime_settings=job.runtime_settings,
        stage_seconds={},
        diagnostics={},
    )
    assert manifest["setups"]["particles"]["count"] == 2
    assert len(manifest["setups"]["particles"]["items"]) == 2
    assert manifest["setups"]["particles"]["order"] == "workbook_row_order"
    assert (
        manifest["setups"]["particle"]["particle_setup_id"]
        == job.particles[0].particle_setup_id
    )
    assert manifest["representation"]["anomaly_definition"] == "particle_minus_clean"

    rows = build_catalog_rows(plan.jobs)
    assert len(rows) == 1
    row = rows[0]
    assert row.n_particles == 2
    assert row.particle_group_id == "group_two"
    assert len(row.particles) == 2
    assert row.particle_x is None  # scalars only for N==1
    assert row.particles[0].center_x == pytest.approx(-5.0)
    assert row.particles[1].center_x == pytest.approx(5.0)


def test_empty_particle_group_hard_errors(tmp_path: Path):
    frames = example_workbook_frames()
    sequences = frames["sequences"].copy()
    sequences["particle_group_id"] = "missing_group"
    frames["sequences"] = sequences
    if "particle_group_id" not in frames["particles"].columns:
        frames["particles"] = frames["particles"].assign(particle_group_id=None)
    path = write_generation_workbook_frames(tmp_path / "empty.xlsx", frames)
    with pytest.raises(Exception, match="empty particle_group_id"):
        validate_generation_plan(
            load_generation_workbook(path),
            repo_root=REPO_ROOT,
        )


def test_overlapping_particle_group_hard_errors(tmp_path: Path):
    from gummybear.particles import ParticleOverlapError

    with pytest.raises(ParticleOverlapError, match="overlapping spheres"):
        attach_particle_group(
            example_workbook_frames(),
            sequence_id="bear_m6_smoke_001",
            particle_group_id="overlap_group",
            centers=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            radius=1.0,
        )

    # Bypass workbook helper validation, then planning must still reject.
    frames = attach_particle_group(
        example_workbook_frames(),
        sequence_id="bear_m6_smoke_001",
        particle_group_id="overlap_group",
        centers=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        radius=1.0,
        require_non_overlapping=False,
    )
    path = write_generation_workbook_frames(tmp_path / "overlap.xlsx", frames)
    with pytest.raises(Exception, match="overlapping spheres"):
        validate_generation_plan(
            load_generation_workbook(path),
            repo_root=REPO_ROOT,
        )


def test_particles_for_job_builds_multi_particle_set(tmp_path: Path):
    frames = attach_particle_group(
        example_workbook_frames(),
        sequence_id="bear_m6_smoke_001",
        particle_group_id="group_two",
        centers=[
            (-5.0, 0.5, 2.5),
            (5.0, -0.5, 2.5),
        ],
    )
    path = write_generation_workbook_frames(tmp_path / "two.xlsx", frames)
    job = validate_generation_plan(
        load_generation_workbook(path),
        repo_root=REPO_ROOT,
    ).jobs[0]
    backend = DefaultSmokePhysicsBackend()
    particle_set = backend._particles_for_job(job)
    assert len(particle_set) == 2
    assert particle_set[0].particle_id == job.particles[0].particle_setup_id
    assert particle_set.metadata["particle_count"] == 2
