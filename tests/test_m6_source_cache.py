"""Phase 4 tests for persistent clean and particle source caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gummybear.datasets.generation_plan import (
    build_execution_plan,
    run_generation_plan,
    validate_generation_plan,
)
from gummybear.datasets.generation_workbook import load_generation_workbook
from gummybear.datasets.sequence_generation import CapturedFrame
from gummybear.datasets.source_cache import (
    CLEAN_REQUIRED_ARRAYS,
    PARTICLE_REQUIRED_ARRAYS,
    CacheEvent,
    SourceCacheStore,
)
from gummybear_validation.milestone_06 import validate_generated_sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_generation_plan.xlsx"


class _FakeDiffusionMesh:
    n_nodes = 5
    n_tets = 2

    def content_hash(self) -> str:
        return "mesh-content-a"


class CacheableFakeBackend:
    def __init__(self) -> None:
        self.context_calls = 0
        self.clean_compute_calls = 0
        self.particle_compute_calls = 0
        self.clean_restore_calls = 0
        self.particle_restore_calls = 0

    def prepare_clean_context(self, job, settings):
        self.context_calls += 1
        return type("Context", (), {"diff_mesh": _FakeDiffusionMesh()})()

    def compute_clean(self, job, context, settings):
        self.clean_compute_calls += 1
        return {"clean": "computed", "context": context}

    def prepare_clean(self, job, settings):
        context = self.prepare_clean_context(job, settings)
        return self.compute_clean(job, context, settings)

    def serialize_clean(self, state):
        arrays = {
            name: np.asarray([index], dtype=float)
            for index, name in enumerate(CLEAN_REQUIRED_ARRAYS)
        }
        return arrays, {"kind": "clean"}

    def restore_clean(self, context, arrays, metadata):
        self.clean_restore_calls += 1
        return {"clean": "restored", "context": context}

    def prepare_particle(self, job, clean_state, settings):
        self.particle_compute_calls += 1
        return {"particle": "computed"}

    def serialize_particle(self, state):
        arrays = {
            name: np.asarray([index], dtype=float)
            for index, name in enumerate(PARTICLE_REQUIRED_ARRAYS)
        }
        return arrays, {"kind": "particle"}

    def restore_particle(self, job, arrays, metadata):
        self.particle_restore_calls += 1
        return {"particle": "restored"}

    def solve_fields(self, job, clean_state, particle_state, settings):
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
        shape = (pose.resolution_y, pose.resolution_x)
        clean = np.full(shape, 1.0)
        particle = np.full(shape, 1.1)
        return CapturedFrame(
            clean=clean,
            particle=particle,
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
        return {"fake_backend": True}


def _execution_plan(cache_root: Path):
    workbook = load_generation_workbook(WORKBOOK)
    plan = validate_generation_plan(workbook, repo_root=REPO_ROOT)
    return build_execution_plan(plan, limit=1, cache_root=cache_root)


def test_second_identical_run_reuses_clean_and_particle_payloads(tmp_path: Path):
    cache_root = tmp_path / "cache"
    output_root = tmp_path / "output"
    backend = CacheableFakeBackend()
    execution = _execution_plan(cache_root)

    first = run_generation_plan(
        execution,
        output_root=output_root,
        physics_backend=backend,
        verbose=True,
    ).generated[0]
    assert first.clean_cache.status == "miss"
    assert first.clean_cache.reason == "miss_not_found"
    assert first.particle_cache.status == "miss"
    assert first.particle_cache.reason == "miss_not_found"
    assert backend.clean_compute_calls == 1
    assert backend.particle_compute_calls == 1

    second = run_generation_plan(
        execution,
        output_root=tmp_path / "output_second",
        physics_backend=backend,
        verbose=True,
    ).generated[0]
    assert second.clean_cache.status == "hit"
    assert second.clean_cache.reason == "hit"
    assert second.particle_cache.status == "hit"
    assert second.particle_cache.reason == "hit"
    assert backend.context_calls == 2
    assert backend.clean_compute_calls == 1
    assert backend.particle_compute_calls == 1
    assert backend.clean_restore_calls == 1
    assert backend.particle_restore_calls == 1

    manifest = json.loads(
        (Path(second.output_path) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["caches"]["persistent_cache_used"] is True
    assert manifest["caches"]["events"]["clean_optical"]["status"] == "hit"
    assert manifest["caches"]["events"]["particle_source"]["status"] == "hit"
    assert "payload_path" not in manifest["caches"]["events"]["clean_optical"]
    particle_sidecar = json.loads(
        Path(second.particle_cache.sidecar_path).read_text(encoding="utf-8")
    )
    assert particle_sidecar["parent_clean_cache_id"] == second.clean_optical_cache_id
    validation = validate_generated_sequence(second.output_path)
    assert validation.ok, validation.errors


def test_force_recompute_replaces_both_payloads(tmp_path: Path):
    backend = CacheableFakeBackend()
    execution = _execution_plan(tmp_path / "cache")
    run_generation_plan(
        execution,
        output_root=tmp_path / "output",
        physics_backend=backend,
        verbose=True,
    )

    forced = run_generation_plan(
        execution,
        output_root=tmp_path / "output_forced",
        physics_backend=backend,
        force_recompute=True,
        verbose=True,
    ).generated[0]
    assert forced.clean_cache.reason == "forced_recompute"
    assert forced.particle_cache.reason == "forced_recompute"
    assert backend.clean_compute_calls == 2
    assert backend.particle_compute_calls == 2


def test_cache_can_be_explicitly_disabled_for_historical_smoke_run(
    tmp_path: Path,
):
    cache_root = tmp_path / "cache"
    backend = CacheableFakeBackend()
    execution = _execution_plan(cache_root)

    results = [
        run_generation_plan(
            execution,
            output_root=tmp_path / f"output_{index}",
            physics_backend=backend,
            use_persistent_cache=False,
            verbose=True,
        ).generated[0]
        for index in range(2)
    ]

    assert [result.clean_cache.status for result in results] == [
        "disabled",
        "disabled",
    ]
    assert [result.particle_cache.status for result in results] == [
        "disabled",
        "disabled",
    ]
    assert backend.clean_compute_calls == 2
    assert backend.particle_compute_calls == 2
    assert not cache_root.exists()
    manifest = json.loads(
        (Path(results[-1].output_path) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["caches"]["persistent_cache_used"] is False


def test_mesh_alignment_mismatch_is_a_structured_miss(tmp_path: Path):
    store = SourceCacheStore(tmp_path)
    event = CacheEvent(
        kind="clean_optical",
        cache_id="a" * 64,
        status="miss",
        reason="miss_not_found",
        payload_path=None,
        sidecar_path=None,
    )
    written = store.write(
        kind="clean_optical",
        cache_id=event.cache_id,
        key_payload={"algorithm_version": "test-v1"},
        payload_schema_version="test-schema-v1",
        arrays={"values": np.asarray([1.0])},
        payload_metadata={},
        mesh_identity={"content_hash": "mesh-a", "num_nodes": 4, "num_tets": 1},
        workbook_provenance={},
        prior_event=event,
    )
    result = store.load(
        kind="clean_optical",
        cache_id=event.cache_id,
        key_payload={"algorithm_version": "test-v1"},
        payload_schema_version="test-schema-v1",
        required_arrays=("values",),
        mesh_identity={"content_hash": "mesh-b", "num_nodes": 4, "num_tets": 1},
    )
    assert written.write_seconds >= 0.0
    assert result.event.status == "miss"
    assert result.event.reason == "miss_mesh_alignment_mismatch"


def test_incomplete_pair_is_never_a_hit(tmp_path: Path):
    cache_id = "b" * 64
    directory = tmp_path / "particle_source"
    directory.mkdir(parents=True)
    np.savez_compressed(directory / f"{cache_id}.npz", values=np.asarray([1.0]))

    result = SourceCacheStore(tmp_path).load(
        kind="particle_source",
        cache_id=cache_id,
        key_payload={"algorithm_version": "test-v1"},
        payload_schema_version="test-schema-v1",
        required_arrays=("values",),
        mesh_identity=None,
    )
    assert result.event.status == "miss"
    assert result.event.reason == "miss_incomplete"
