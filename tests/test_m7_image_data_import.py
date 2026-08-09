"""M7.1 catalog membership checks against the installed validation workbook."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import pytest

from tomography_ml.gummybear_data_catalog import (
    catalog_jobs_to_dataframe,
    load_catalog_jobs,
    load_catalog_plan,
)


VALIDATION_ROOT = files("tomography_ml_validation") / "test_data"


@pytest.fixture(scope="module")
def validation_root():
    """Resolved installed validation ``test_data`` directory."""
    with as_file(VALIDATION_ROOT) as root:
        yield Path(root)


@pytest.fixture(scope="module")
def plan(validation_root: Path):
    workbook_path = validation_root / "configs" / "m6" / "m6_matrix_plan.xlsx"
    return load_catalog_plan(
        workbook_path=workbook_path,
        root_path=validation_root,
    )


@pytest.fixture(scope="module")
def catalog_df(validation_root: Path, plan):
    workbook_path = validation_root / "configs" / "m6" / "m6_matrix_plan.xlsx"
    jobs = load_catalog_jobs(
        workbook_path=workbook_path,
        root_path=validation_root,
    )
    # Keep plan/jobs consistent even if load_catalog_jobs is reimplemented.
    assert [job.sequence_id for job in jobs] == [
        job.sequence_id for job in plan.jobs
    ]
    return catalog_jobs_to_dataframe(jobs)


def test_catalog_has_one_row_per_plan_job(catalog_df, plan):
    assert len(catalog_df) == len(plan.jobs)


def test_sample_ids_are_sequential(catalog_df):
    assert catalog_df["sample_id"].tolist() == list(range(len(catalog_df)))


def test_sequence_ids_match_plan_jobs(catalog_df, plan):
    expected = [job.sequence_id for job in plan.jobs]
    assert catalog_df["sequence_id"].tolist() == expected


def test_disabled_sequence_ids_not_in_catalog(catalog_df, plan):
    catalog_ids = set(catalog_df["sequence_id"])
    disabled_ids = set(plan.disabled_sequence_ids)
    assert catalog_ids.isdisjoint(disabled_ids)


def test_selected_status_is_workbook_enabled(catalog_df):
    assert set(catalog_df["selected_status"]) == {"workbook_enabled"}


def test_catalog_ignores_extra_generated_directories(catalog_df, validation_root: Path):
    """Membership is workbook/plan driven, not directory discovery."""
    generated = validation_root / "data" / "generated" / "m6_5"
    assert generated.is_dir()

    on_disk_sequence_dirs = {
        path.name
        for path in generated.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }
    catalog_ids = set(catalog_df["sequence_id"])

    # Fixture data currently includes disabled/orphan-like dirs such as
    # bear_m6_matrix_004. Those must not become catalog rows.
    extras_on_disk = on_disk_sequence_dirs - catalog_ids
    assert extras_on_disk, (
        "Expected at least one non-catalog generated directory in the "
        "installed fixture so membership is not vacuously filesystem-equal."
    )
    assert catalog_ids.isdisjoint(extras_on_disk)
