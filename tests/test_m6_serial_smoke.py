"""Phase 2 tests for serial orchestration, role writing, and manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gummybear.datasets.generation_plan import (
    GenerationPlanError,
    build_execution_plan,
    run_generation_plan,
    validate_generation_plan,
)
from gummybear.datasets.generation_workbook import load_generation_workbook
from gummybear.datasets.role_images import orient_camera_image_for_storage
from gummybear.datasets.sequence_generation import CapturedFrame
from gummybear.datasets.sequence_writer import frame_filename


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_generation_plan.xlsx"


class FakeSmokeBackend:
    def __init__(self) -> None:
        self.clean_calls = 0
        self.particle_calls = 0
        self.solve_calls = 0
        self.capture_calls = 0

    def prepare_clean(self, job, settings):
        self.clean_calls += 1
        return {"sequence_id": job.sequence_id}

    def prepare_particle(self, job, clean_state, settings):
        self.particle_calls += 1
        return {"clean": clean_state}

    def solve_fields(self, job, clean_state, particle_state, settings):
        self.solve_calls += 1
        return {"solved": True}

    def capture_frame(
        self,
        job,
        pose,
        clean_state,
        particle_state,
        field_state,
        settings,
    ):
        self.capture_calls += 1
        shape = (pose.resolution_y, pose.resolution_x)
        clean = np.full(shape, 1.0 + 0.1 * pose.frame_index)
        particle = clean.copy()
        particle[
            shape[0] // 3 : 2 * shape[0] // 3, shape[1] // 3 : 2 * shape[1] // 3
        ] += 0.25
        return CapturedFrame(
            clean=clean,
            particle=particle,
            metadata={
                "frame_index": pose.frame_index,
                "angle_deg": pose.angle_deg,
                "camera_position": [float(pose.frame_index), -80.0, 2.5],
                "look_at": [-1.1180591583251953, 0.4537315368652344, 2.5],
                "up": [0.0, 0.0, 1.0],
                "camera_kind": pose.camera_kind,
                "resolution": [pose.resolution_y, pose.resolution_x],
            },
        )

    def diagnostics(self, clean_state, particle_state, field_state):
        return {"fake_backend": True}


def _execution_plan():
    workbook = load_generation_workbook(WORKBOOK)
    plan = validate_generation_plan(workbook, repo_root=REPO_ROOT)
    return build_execution_plan(plan, limit=1)


def test_serial_smoke_writes_aligned_roles_and_manifest(tmp_path: Path):
    backend = FakeSmokeBackend()
    result = run_generation_plan(
        _execution_plan(),
        output_root=tmp_path,
        max_workers=1,
        physics_backend=backend,
        verbose=True,
    )

    assert len(result.generated) == 1
    generated = result.generated[0]
    assert generated.frame_count == 6
    assert backend.clean_calls == 1
    assert backend.particle_calls == 1
    assert backend.solve_calls == 1
    assert backend.capture_calls == 6

    sequence_dir = Path(generated.output_path)
    clean_names = sorted(path.name for path in (sequence_dir / "clean").glob("*.jpg"))
    particle_names = sorted(
        path.name for path in (sequence_dir / "particle").glob("*.jpg")
    )
    observed_names = sorted(
        path.name for path in (sequence_dir / "observed").glob("*.jpg")
    )
    anomaly_names = sorted(
        path.name for path in (sequence_dir / "anomaly").glob("*.png")
    )
    anomaly_raw_names = sorted(
        path.name for path in (sequence_dir / "anomaly").glob("*.raw.tif")
    )
    assert clean_names == particle_names == observed_names
    assert len(clean_names) == len(anomaly_names) == len(anomaly_raw_names) == 6
    assert [
        name.replace(".png", ".raw.tif") for name in anomaly_names
    ] == anomaly_raw_names
    assert "_frame_0000_" in clean_names[0]
    assert "_frame_0005_" in clean_names[-1]

    clean_raw_names = sorted(
        path.name for path in (sequence_dir / "clean").glob("*.raw.tif")
    )
    assert [name.replace(".jpg", ".raw.tif") for name in clean_names] == (
        clean_raw_names
    )
    with Image.open(sequence_dir / "clean" / clean_raw_names[0]) as raw_image:
        assert raw_image.mode == "F"
        raw_values = np.asarray(raw_image, dtype=float)
    assert raw_values.shape == (128, 128)
    assert float(raw_values.min()) >= 1.0

    for particle_name, observed_name in zip(
        particle_names,
        observed_names,
        strict=True,
    ):
        particle_bytes = (sequence_dir / "particle" / particle_name).read_bytes()
        observed_bytes = (sequence_dir / "observed" / observed_name).read_bytes()
        assert particle_bytes == observed_bytes
        particle_raw = particle_name.replace(".jpg", ".raw.tif")
        observed_raw = observed_name.replace(".jpg", ".raw.tif")
        assert np.array_equal(
            np.asarray(Image.open(sequence_dir / "particle" / particle_raw)),
            np.asarray(Image.open(sequence_dir / "observed" / observed_raw)),
        )

    with Image.open(sequence_dir / "clean" / clean_names[0]) as image:
        assert image.size == (128, 128)
        assert image.mode == "L"

    manifest = json.loads((sequence_dir / "manifest.json").read_text())
    assert manifest["representation"]["raw_float_sidecar"]["extension"] == "raw.tif"
    assert manifest["frames"][0]["filenames"]["clean"].endswith(".jpg")
    assert manifest["frames"][0]["filenames"]["clean_raw"].endswith(".raw.tif")
    assert manifest["frames"][0]["filenames"]["anomaly"].endswith(".png")
    assert manifest["frames"][0]["filenames"]["anomaly_raw"].endswith(".raw.tif")
    assert "anomaly" in manifest["representation"]["raw_float_sidecar"]["roles"]
    assert manifest["schema_version"] == "1.6-m6-draft"
    assert manifest["generator_version"] == "m6.5-draft"
    assert "split" not in manifest
    assert "seed" not in manifest
    assert "split" not in manifest["resolved_job"]["sequence"]
    assert "seed" not in manifest["resolved_job"]["sequence"]
    assert manifest["resolved_job_hash"] == _execution_plan().jobs[0].resolved_job_hash
    assert manifest["resolved_job"]["sequence"]["sequence_id"] == ("bear_m6_smoke_001")
    assert manifest["workbook"]["workbook_path"] == ("configs/m6/m6_generation_plan.xlsx")
    assert "path" not in manifest["workbook"]
    assert manifest["phantom"]["stl_path"] == "cad/proto_bear.stl"
    assert manifest["representation"]["image_domain"] == "camera_intensity"
    assert (
        manifest["representation"]["composition_domain"]
        == "linear_camera_intensity_before_jpeg"
    )
    assert manifest["representation"]["anomaly_definition"] == "particle_minus_clean"
    assert manifest["representation"]["pixel_orientation"] == {
        "camera_up": "image_top",
        "transform_from_camera_sample_grid": "flip_axis_0",
    }
    assert manifest["caches"]["diffusion_operator_cache"] is None
    assert manifest["setups"]["optical"]["optical_setup_id"]
    assert manifest["setups"]["particle"]["particle_setup_id"]
    assert manifest["setups"]["particles"]["count"] == 1
    assert len(manifest["setups"]["particles"]["items"]) == 1
    assert manifest["setups"]["diffusion"]["diffusion_setup_id"]
    expected_setup_sheets = {
        "optical": "optical_setups",
        "particle": "particles",
        "diffusion": "diffusion_setups",
        "camera": "camera_schedules",
        "corruption": "corruptions",
    }
    for setup_name, workbook_sheet in expected_setup_sheets.items():
        setup = manifest["setups"][setup_name]
        assert setup["workbook_name"] == "m6_generation_plan.xlsx"
        assert setup["workbook_sheet"] == workbook_sheet
        assert setup["source_excel_row"] == 2
    assert manifest["setups"]["particles"]["items"][0]["source_excel_row"] == 2
    assert len(manifest["frames"]) == 6
    assert [frame["frame_index"] for frame in manifest["frames"]] == list(range(6))


def test_generation_progress_prints_done_over_total(tmp_path: Path, capsys):
    backend = FakeSmokeBackend()
    result = run_generation_plan(
        _execution_plan(),
        output_root=tmp_path,
        max_workers=1,
        physics_backend=backend,
        progress=True,
        verbose=True,
    )
    assert len(result.generated) == 1
    assert capsys.readouterr().out.strip().splitlines() == ["1/1"]


def test_second_identical_run_is_output_noop(tmp_path: Path):
    backend = FakeSmokeBackend()
    execution = _execution_plan()

    first = run_generation_plan(
        execution,
        output_root=tmp_path,
        physics_backend=backend,
        verbose=True,
    )
    counts_after_first = (
        backend.clean_calls,
        backend.particle_calls,
        backend.solve_calls,
        backend.capture_calls,
    )
    second = run_generation_plan(
        execution,
        output_root=tmp_path,
        physics_backend=backend,
        verbose=True,
    )

    assert len(first.generated) == 1
    assert second.generated == ()
    assert second.skipped == ("bear_m6_smoke_001",)
    assert [
        item.status
        for item in second.output_items
        if item.sequence_id == "bear_m6_smoke_001"
    ] == ["output_complete_current"]
    assert (
        backend.clean_calls,
        backend.particle_calls,
        backend.solve_calls,
        backend.capture_calls,
    ) == counts_after_first


def test_serial_baseline_rejects_max_workers_without_parallel():
    with pytest.raises(GenerationPlanError, match="requires parallel=True"):
        run_generation_plan(_execution_plan(), dry_run=True, max_workers=2,
verbose=True,
)


def test_default_parallel_workers_leaves_two_cores_free(monkeypatch):
    from gummybear.datasets.generation_plan import (
        default_parallel_workers,
        resolve_generation_workers,
    )

    monkeypatch.setattr(
        "gummybear.datasets.generation_plan.os.cpu_count",
        lambda: 12,
    )
    assert default_parallel_workers() == 10
    assert resolve_generation_workers(parallel=True) == 10
    assert resolve_generation_workers(parallel=True, max_workers=3) == 3
    assert resolve_generation_workers(parallel=False) == 1
    monkeypatch.setattr(
        "gummybear.datasets.generation_plan.os.cpu_count",
        lambda: 1,
    )
    assert default_parallel_workers() == 1


def test_parallel_sequence_generation_with_injected_backend(tmp_path: Path):
    from dataclasses import replace
    from threading import Lock
    from time import perf_counter, sleep

    workbook = load_generation_workbook(WORKBOOK)
    plan = validate_generation_plan(workbook, repo_root=REPO_ROOT)
    execution = build_execution_plan(plan, limit=1)
    job = execution.jobs[0]
    second = replace(job, sequence_id=f"{job.sequence_id}_b")
    third = replace(job, sequence_id=f"{job.sequence_id}_c")
    multi = replace(execution, jobs=(job, second, third))

    class ThreadSafeFakeBackend(FakeSmokeBackend):
        def __init__(self) -> None:
            super().__init__()
            self._lock = Lock()
            self.clean_started: list[tuple[str, float]] = []
            self.clean_finished: list[tuple[str, float]] = []

        def prepare_clean(self, job, settings):
            with self._lock:
                self.clean_started.append((job.sequence_id, perf_counter()))
            sleep(0.05)
            with self._lock:
                result = super().prepare_clean(job, settings)
                self.clean_finished.append((job.sequence_id, perf_counter()))
                return result

        def prepare_particle(self, job, clean_state, settings):
            with self._lock:
                return super().prepare_particle(job, clean_state, settings)

        def solve_fields(self, job, clean_state, particle_state, settings):
            with self._lock:
                return super().solve_fields(
                    job, clean_state, particle_state, settings
                )

        def capture_frame(self, job, pose, clean_state, particle_state, field_state, settings):
            with self._lock:
                return super().capture_frame(
                    job,
                    pose,
                    clean_state,
                    particle_state,
                    field_state,
                    settings,
                )

    backend = ThreadSafeFakeBackend()
    # jobs_to_generate is sorted by sequence_id inside run_generation_plan
    ordered_ids = sorted([job.sequence_id, second.sequence_id, third.sequence_id])
    result = run_generation_plan(
        multi,
        output_root=tmp_path,
        parallel=True,
        max_workers=2,
        physics_backend=backend,
        use_persistent_cache=False,
        verbose=True,
    )
    assert sorted(item.sequence_id for item in result.generated) == ordered_ids
    assert backend.clean_calls == 3
    assert [seq for seq, _ in backend.clean_started][0] == ordered_ids[0]
    first_finished = next(
        t for seq, t in backend.clean_finished if seq == ordered_ids[0]
    )
    later_starts = [
        t for seq, t in backend.clean_started if seq != ordered_ids[0]
    ]
    assert later_starts
    assert all(start >= first_finished for start in later_starts)
    manifests = [
        json.loads((Path(item.output_path) / "manifest.json").read_text())
        for item in result.generated
    ]
    assert {manifest["generation"]["max_workers"] for manifest in manifests} == {2}


def test_frame_filename_sorts_by_zero_padded_index():
    names = [
        frame_filename("sequence", index, 300.0 - index, extension="jpg")
        for index in range(12)
    ]
    assert sorted(reversed(names)) == names


def test_camera_images_are_flipped_vertically_for_storage():
    camera_grid = np.array([[1, 2], [3, 4]])
    assert orient_camera_image_for_storage(camera_grid).tolist() == [
        [3, 4],
        [1, 2],
    ]
