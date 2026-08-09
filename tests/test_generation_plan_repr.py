"""Tests for GenerationPlan readable __repr__."""

from __future__ import annotations

from pathlib import Path

from gummybear.datasets.generation_plan import validate_generation_plan
from gummybear.datasets.generation_workbook import load_generation_workbook


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_generation_plan.xlsx"


def test_generation_plan_repr_is_readable_and_repo_relative():
    plan = validate_generation_plan(
        load_generation_workbook(WORKBOOK),
        repo_root=REPO_ROOT,
    )
    text = repr(plan)

    assert "GenerationPlan(" in text
    assert "enabled_jobs=1" in text
    assert "sequence_id" in text
    assert "bear_m6_smoke_001" in text
    assert "opt_smoke_backlight_001" in text
    assert "configs/m6/m6_generation_plan.xlsx" in text
    assert "data/generated/m6_2" in text

    # No absolute home / username leakage from workbook or output roots.
    assert str(Path.home()) not in text
    assert "/Users/" not in text
    assert "workbook_sha256=" in text
    # Hash is truncated in the header line.
    assert len(plan.workbook_sha256) == 64
    assert plan.workbook_sha256 not in text
