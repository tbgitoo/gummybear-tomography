"""STL root can differ from catalog / data root without changing path identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from gummybear.datasets.generation_plan import (
    GenerationPlanError,
    _resolve_stl_sha256,
    validate_generation_plan,
)
from gummybear.datasets.generation_workbook import load_generation_workbook
from gummybear.geometry import sha256_file
from tomography_ml.gummybear_data_catalog import load_catalog_jobs


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPO_ROOT / "configs" / "m6" / "m6_generation_plan.xlsx"
STL_REL = "cad/proto_bear.stl"


def test_resolve_stl_sha256_prefers_stl_root(tmp_path: Path, monkeypatch):
    stl_src = REPO_ROOT / STL_REL
    assert stl_src.is_file()
    expected = sha256_file(stl_src)

    # Data root without cad/; STL lives under a separate absolute root.
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    stl_root = tmp_path / "cad_root"
    (stl_root / "cad").mkdir(parents=True)
    target = stl_root / STL_REL
    target.write_bytes(stl_src.read_bytes())

    # Avoid accidental cwd / repo-relative hits during the negative check.
    monkeypatch.chdir(data_root)

    with pytest.raises(GenerationPlanError, match="STL file not found"):
        _resolve_stl_sha256(STL_REL, repo_root=data_root)

    assert (
        _resolve_stl_sha256(STL_REL, repo_root=data_root, stl_root=stl_root)
        == expected
    )


def test_load_catalog_jobs_accepts_stl_root(tmp_path: Path):
    # Workbook under repo; pretend data root is elsewhere; STL via stl_root=ROOT.
    jobs = load_catalog_jobs(
        WORKBOOK,
        root_path=tmp_path,
        stl_root=REPO_ROOT,
    )
    assert jobs
    assert jobs[0].stl_path == STL_REL
    assert jobs[0].stl_sha256 == sha256_file(REPO_ROOT / STL_REL)


def test_validate_generation_plan_stl_root_keeps_relative_path(tmp_path: Path):
    workbook = load_generation_workbook(WORKBOOK)
    plan = validate_generation_plan(
        workbook,
        repo_root=tmp_path,
        stl_root=REPO_ROOT,
    )
    assert plan.jobs[0].stl_path == STL_REL
