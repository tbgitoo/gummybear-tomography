"""Tests for the workbook-path generation convenience helper."""

from __future__ import annotations

from pathlib import Path

from gummybear.datasets.generation_plan import run_generation_workbook


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_generation_plan.xlsx"


def test_run_generation_workbook_dry_run_from_path(tmp_path: Path):
    result = run_generation_workbook(
        WORKBOOK,
        repo_root=REPO_ROOT,
        output_root=tmp_path / "out",
        dry_run=True,
        use_persistent_cache=False,
        verbose=True,
    )
    assert result.dry_run is True
    assert "bear_m6_smoke_001" in result.skipped
    assert result.generated == ()


def test_run_generation_workbook_accepts_string_paths(tmp_path: Path):
    result = run_generation_workbook(
        str(WORKBOOK),
        repo_root=str(REPO_ROOT),
        output_root=str(tmp_path / "out"),
        sequence_id="bear_m6_smoke_001",
        dry_run=True,
        use_persistent_cache=False,
        verbose=True,
    )
    assert result.dry_run is True
    assert result.skipped == ("bear_m6_smoke_001",)


def test_run_generation_workbook_remove_stale_rejects_dry_run(tmp_path: Path):
    import pytest
    from gummybear.datasets.generation_plan import GenerationPlanError

    with pytest.raises(GenerationPlanError, match="remove_stale"):
        run_generation_workbook(
            WORKBOOK,
            repo_root=REPO_ROOT,
            output_root=tmp_path / "out",
            dry_run=True,
            remove_stale=True,
            use_persistent_cache=False,
            verbose=True,
        )
