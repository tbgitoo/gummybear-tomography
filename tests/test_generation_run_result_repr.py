"""Tests for GenerationRunResult readable __repr__."""

from __future__ import annotations

from pathlib import Path

from gummybear.datasets.generation_plan import run_generation_workbook
from gummybear.datasets.output_plan import OutputDeltaItem
from gummybear.datasets.sequence_generation import (
    GeneratedSequenceResult,
    GenerationRunResult,
)
from gummybear.datasets.source_cache import CacheEvent


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_generation_plan.xlsx"


def test_generation_run_result_repr_dry_run_is_readable_and_repo_relative(
    tmp_path: Path,
):
    result = run_generation_workbook(
        WORKBOOK,
        repo_root=REPO_ROOT,
        output_root=tmp_path / "out",
        dry_run=True,
        use_persistent_cache=False,
        verbose=True,
    )
    text = repr(result)

    assert "GenerationRunResult(" in text
    assert "dry_run=True" in text
    assert "bear_m6_smoke_001" in text
    assert "sequence_id" in text
    assert "missing" in text or "complete_current" in text
    assert str(Path.home()) not in text
    assert "/Users/" not in text


def test_generation_run_result_repr_includes_generated_timing_and_cache():
    result = GenerationRunResult(
        generated=(
            GeneratedSequenceResult(
                sequence_id="seq_a",
                output_path=str(REPO_ROOT / "data" / "generated" / "demo" / "seq_a"),
                frame_count=6,
                clean_optical_cache_id="a" * 64,
                particle_source_cache_id="b" * 64,
                stage_seconds={"prepare": 1.25, "capture": 2.5},
                clean_cache=CacheEvent(
                    kind="clean",
                    cache_id="a" * 64,
                    status="hit",
                    reason="ok",
                    payload_path=None,
                    sidecar_path=None,
                ),
                particle_cache=CacheEvent(
                    kind="particle",
                    cache_id="b" * 64,
                    status="miss",
                    reason="absent",
                    payload_path=None,
                    sidecar_path=None,
                ),
            ),
        ),
        skipped=("seq_b",),
        failed=(),
        dry_run=False,
        output_items=(
            OutputDeltaItem(
                sequence_id="seq_a",
                output_path=str(REPO_ROOT / "data" / "generated" / "demo" / "seq_a"),
                status="output_missing",
                reason="will_generate",
                details={},
            ),
            OutputDeltaItem(
                sequence_id="seq_b",
                output_path=str(REPO_ROOT / "data" / "generated" / "demo" / "seq_b"),
                status="output_complete_current",
                reason="already_current",
                details={},
            ),
        ),
        verbose=True,
    )
    text = repr(result)

    assert "generated=1" in text
    assert "skipped=1" in text
    assert "seq_a" in text
    assert "generated" in text
    assert "hit" in text
    assert "miss" in text
    assert "3.8" in text  # 1.25 + 2.5
    assert "data/generated/demo/seq_a" in text
    assert "cache_ids=" in text
    assert "aaaaaaaaaaaa" in text
    assert "a" * 64 not in text
    assert str(Path.home()) not in text
    assert "/Users/" not in text

def test_generation_run_result_repr_compact_default_shows_cache_computed_orphan():
    result = GenerationRunResult(
        generated=(
            GeneratedSequenceResult(
                sequence_id="seq_a",
                output_path=str(REPO_ROOT / "data" / "generated" / "demo" / "seq_a"),
                frame_count=2,
                clean_optical_cache_id="a" * 64,
                particle_source_cache_id="b" * 64,
                stage_seconds={"prepare": 0.5},
                clean_cache=CacheEvent(
                    kind="clean",
                    cache_id="a" * 64,
                    status="hit",
                    reason="ok",
                    payload_path=None,
                    sidecar_path=None,
                ),
                particle_cache=CacheEvent(
                    kind="particle",
                    cache_id="b" * 64,
                    status="hit",
                    reason="ok",
                    payload_path=None,
                    sidecar_path=None,
                ),
            ),
        ),
        skipped=("seq_b",),
        failed=(),
        dry_run=False,
        output_items=(
            OutputDeltaItem(
                sequence_id="seq_a",
                output_path=str(REPO_ROOT / "data" / "generated" / "demo" / "seq_a"),
                status="output_missing",
                reason="will_generate",
                details={},
            ),
            OutputDeltaItem(
                sequence_id="seq_b",
                output_path=str(REPO_ROOT / "data" / "generated" / "demo" / "seq_b"),
                status="output_complete_current",
                reason="already_current",
                details={},
            ),
            OutputDeltaItem(
                sequence_id="seq_orphan",
                output_path=str(REPO_ROOT / "data" / "generated" / "demo" / "seq_orphan"),
                status="output_orphaned_not_requested",
                reason="not_in_workbook",
                details={},
            ),
            OutputDeltaItem(
                sequence_id="seq_disabled",
                output_path=str(REPO_ROOT / "data" / "generated" / "demo" / "seq_disabled"),
                status="output_disabled_not_run",
                reason="disabled",
                details={},
            ),
        ),
        verbose=False,
    )
    text = repr(result)

    assert "jobs=" in text
    assert "seq_a" in text and "computed" in text
    assert "seq_b" in text and "cache" in text
    assert "seq_orphan" in text and "orphaned" in text
    assert "disabled=1" in text
    assert "seq_disabled" in text and "disabled" in text
    assert "disabled_rows=(none)" not in text
    assert "cache_ids=" not in text
    assert "sequences=" not in text


def test_generation_run_result_repr_reports_no_disabled_rows_when_absent():
    result = GenerationRunResult(
        generated=(),
        skipped=("seq_b",),
        failed=(),
        dry_run=False,
        output_items=(
            OutputDeltaItem(
                sequence_id="seq_b",
                output_path=str(REPO_ROOT / "data" / "generated" / "demo" / "seq_b"),
                status="output_complete_current",
                reason="already_current",
                details={},
            ),
        ),
        verbose=False,
    )
    text = repr(result)
    assert "disabled=0" in text
    assert "disabled_rows=(none)" in text

    verbose = GenerationRunResult(
        generated=result.generated,
        skipped=result.skipped,
        failed=result.failed,
        dry_run=result.dry_run,
        output_items=result.output_items,
        verbose=True,
    )
    verbose_text = repr(verbose)
    assert "disabled=0" in verbose_text
    assert "disabled_rows=(none)" in verbose_text


def test_run_generation_workbook_default_verbose_is_false(tmp_path: Path):
    result = run_generation_workbook(
        WORKBOOK,
        repo_root=REPO_ROOT,
        output_root=tmp_path / "out",
        dry_run=True,
        use_persistent_cache=False,
    )
    assert result.verbose is False
    text = repr(result)
    assert "jobs=" in text
    assert "cache_ids=" not in text

