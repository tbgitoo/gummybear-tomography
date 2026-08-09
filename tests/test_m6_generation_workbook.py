"""Phase 1 tests for M6 generation workbook parsing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gummybear.datasets.generation_workbook import (
    REQUIRED_SHEETS,
    WorkbookValidationError,
    load_generation_workbook,
    write_example_generation_workbook,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_generation_plan.xlsx"


@pytest.fixture(scope="module")
def example_workbook_path() -> Path:
    if not EXAMPLE_WORKBOOK.is_file():
        write_example_generation_workbook(EXAMPLE_WORKBOOK)
    return EXAMPLE_WORKBOOK


def test_example_workbook_loads(example_workbook_path: Path):
    workbook = load_generation_workbook(example_workbook_path)
    assert workbook.path == example_workbook_path.resolve()
    assert len(workbook.sha256) == 64
    assert workbook.sheet_names == REQUIRED_SHEETS
    for sheet in REQUIRED_SHEETS:
        assert sheet in workbook.sheets
    assert "source_aperture" not in workbook.rows("optical_setups")[0].values
    assert "source_intensity" in workbook.rows("optical_setups")[0].values
    assert workbook.rows("optical_setups")[0].values["source_intensity"] == 1.0
    assert "alpha_direct" not in workbook.rows("optical_setups")[0].values
    assert "g" not in workbook.rows("optical_setups")[0].values
    assert "mu_a" not in workbook.rows("diffusion_setups")[0].values
    assert "D" not in workbook.rows("diffusion_setups")[0].values
    assert "g" in workbook.rows("diffusion_setups")[0].values
    assert "alpha_direct" in workbook.rows("diffusion_setups")[0].values
    assert workbook.rows("diffusion_setups")[0].values["alpha_direct"] == 0.0
    assert workbook.rows("diffusion_setups")[0].values["g"] == 0.0


def test_disabled_rows_are_ignored_for_enabled_helpers(
    example_workbook_path: Path,
):
    workbook = load_generation_workbook(example_workbook_path)
    enabled_sequences = workbook.enabled_rows("sequences")
    assert len(enabled_sequences) == 1
    assert enabled_sequences[0].setup_id == "bear_m6_smoke_001"

    corruption_rows = workbook.rows("corruptions")
    assert any(row.setup_id == "none" for row in corruption_rows)
    assert workbook.enabled_rows("corruptions") == ()
    assert all(not row.enabled for row in corruption_rows)


def test_missing_required_sheet_fails(tmp_path: Path, example_workbook_path: Path):
    frames = pd.read_excel(
        example_workbook_path,
        sheet_name=None,
        engine="openpyxl",
    )
    del frames["particles"]
    broken = tmp_path / "missing_sheet.xlsx"
    with pd.ExcelWriter(broken, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    with pytest.raises(WorkbookValidationError, match="particles"):
        load_generation_workbook(broken)


def test_missing_required_column_fails(tmp_path: Path, example_workbook_path: Path):
    frames = pd.read_excel(
        example_workbook_path,
        sheet_name=None,
        engine="openpyxl",
    )
    frames["sequences"] = frames["sequences"].drop(columns=["seed"])
    broken = tmp_path / "missing_column.xlsx"
    with pd.ExcelWriter(broken, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    with pytest.raises(WorkbookValidationError, match="seed"):
        load_generation_workbook(broken)
