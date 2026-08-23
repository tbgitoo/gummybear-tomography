"""Notebook-facing helpers for Milestone 7 catalog / task-dataset demos."""

from __future__ import annotations

from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from gummybear.datasets.generation_plan import validate_generation_plan
from gummybear.datasets.generation_workbook import (
    attach_particle_group,
    example_workbook_frames,
    load_generation_workbook,
    write_generation_workbook_frames,
)
from tomography_ml.gummybear_data_catalog import (
    DatasetTaskSpec,
    build_catalog_rows,
    build_task_dataset,
)
from tomography_ml.gummybear_data_catalog.catalog import CatalogRow
from tomography_ml.gummybear_data_catalog.task_dataset import M7_IMAGE_REPRESENTATION

CATALOG_JOB_DISPLAY_COLUMNS = (
    "sample_id",
    "sequence_id",
    "split",
    "particle_setup_id",
    "particle_group_id",
    "n_particles",
    "output_root",
    "sequence_dir",
    "camera_schedule_id",
    "diffusion_setup_id",
    "selected_status",
    "disabled_reported",
)

RECONCILIATION_DISPLAY_COLUMNS = (
    "sample_id",
    "sequence_id",
    "sequence_dir_exists",
    "manifest_exists",
    "field_status",
)

SCHEDULE_IDENTITY_DISPLAY_COLUMNS = (
    "sequence_id",
    "camera_schedule_id",
    "frame_count",
    "resolution_x",
    "resolution_y",
    "first_angle_deg",
    "last_angle_deg",
    "schedule_status",
)

FLAT_CATALOG_DISPLAY_COLUMNS = (
    "sample_id",
    "sequence_id",
    "camera_schedule_id",
    "frame_count",
    "n_particles",
    "particle_group_id",
    "particle_radius",
    "diffusion_setup_id",
    "schema_version",
    "composition_domain",
    "field_status",
)


def validation_fixture_paths(
    *,
    schedule_workbook: str = "configs/m6/m6_matrix_plan.xlsx",
    output_scenario: str = "data/generated/m6_5",
) -> dict[str, Path]:
    """Packaged ``tomography_ml_validation`` test_data paths for M7 notebooks."""
    root = files("tomography_ml_validation") / "test_data"
    output_root = root / output_scenario
    return {
        "validation_root": Path(str(root)),
        "workbook_path": Path(str(root / schedule_workbook)),
        "output_root": Path(str(output_root)),
        "cache_root": Path(str(output_root / "_cache")),
    }


def select_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a dataframe restricted to *columns* that exist."""
    present = [name for name in columns if name in df.columns]
    return df[present]


def catalog_rows_dataframe(
    rows: Sequence[CatalogRow],
    *,
    columns: Sequence[str] | None = FLAT_CATALOG_DISPLAY_COLUMNS,
) -> pd.DataFrame:
    """Flatten :class:`CatalogRow` objects for pandas display."""
    frame = pd.DataFrame(asdict(row) for row in rows)
    if columns is None:
        return frame
    return select_columns(frame, columns)


def write_demo_multi_particle_workbook(
    path: Path | str,
    *,
    sequence_id: str = "bear_m6_smoke_001",
    particle_group_id: str,
    centers: Sequence[tuple[float, float, float]],
    repo_root: Path | str,
) -> Path:
    """Write a temporary two-sphere workbook for catalog-label demos (no FEM)."""
    path = Path(path)
    frames = attach_particle_group(
        example_workbook_frames(),
        sequence_id=sequence_id,
        particle_group_id=particle_group_id,
        centers=list(centers),
    )
    write_generation_workbook_frames(path, frames)
    return path


def build_demo_multi_particle_catalog_rows(
    path: Path | str,
    *,
    repo_root: Path | str,
) -> tuple[CatalogRow, ...]:
    """Validate a demo workbook and return flat catalog rows."""
    jobs = validate_generation_plan(
        load_generation_workbook(path),
        repo_root=repo_root,
    ).jobs
    return build_catalog_rows(jobs)


def particle_labels_dataframe(row: CatalogRow) -> pd.DataFrame:
    """Expand ordered ``row.particles`` for notebook display."""
    return pd.DataFrame(
        [
            {
                "sequence_id": row.sequence_id,
                "n_particles": row.n_particles,
                "particle_group_id": row.particle_group_id,
                "particle_setup_id": label.particle_setup_id,
                "center_x": label.center_x,
                "center_y": label.center_y,
                "center_z": label.center_z,
                "radius": label.radius,
            }
            for label in row.particles
        ]
    )


def assert_single_particle_corpus(rows: Sequence[CatalogRow]) -> None:
    """Installed matrix corpus remains one sphere per sequence."""
    for row in rows:
        if row.n_particles != 1:
            raise AssertionError(
                f"expected n_particles==1 for {row.sequence_id!r}, got {row.n_particles}"
            )
        if row.particle_x is None or row.particle_radius is None:
            raise AssertionError(
                f"expected compatibility scalars for {row.sequence_id!r}"
            )


def assert_single_particle_flat_labels(rows: Sequence[CatalogRow]) -> None:
    """Scalars match ``particles[0]`` when ``n_particles == 1``."""
    for row in rows:
        if row.n_particles != 1:
            continue
        if row.particles[0].radius != row.particle_radius:
            raise AssertionError(
                f"particle_radius mismatch for {row.sequence_id!r}"
            )


def probe_observed_image_shapes(
    catalog_rows: Sequence[CatalogRow],
    *,
    image_representation: str = M7_IMAGE_REPRESENTATION,
) -> dict[str, tuple[int, ...]]:
    """Lazy-load ``observed_ref`` shapes via a tiny probe task (complete rows only)."""
    probe = build_task_dataset(
        catalog_rows,
        DatasetTaskSpec(
            name="shape_probe",
            row_filter={"field_status": "complete"},
            x_fields=("observed_ref",),
            y_fields=("sequence_id",),
            image_representation=image_representation,
        ),
    )
    shapes: dict[str, tuple[int, ...]] = {}
    for index in range(len(probe)):
        x, y = probe[index]
        shapes[str(y["sequence_id"])] = tuple(x["observed_ref"].shape)
    return shapes


def build_catalog_status_table(
    catalog_rows: Sequence[CatalogRow],
    subset_rows: Sequence[CatalogRow],
    *,
    image_shape_by_sequence_id: Mapping[str, tuple[int, ...]] | None = None,
    ready_task_views: Sequence[str] = ("inpainting", "particle_localization"),
) -> pd.DataFrame:
    """Schedule-aware readiness view for M7.6."""
    subset_ids = {row.sequence_id for row in subset_rows}
    image_shapes = dict(image_shape_by_sequence_id or {})
    records: list[dict[str, Any]] = []
    for row in catalog_rows:
        in_subset = row.sequence_id in subset_ids
        image_shape = image_shapes.get(row.sequence_id)
        if row.field_status != "complete":
            catalog_status = "incomplete_artifact"
            candidate_views: list[str] = []
        elif not in_subset:
            catalog_status = "excluded_by_schedule"
            candidate_views = []
        else:
            catalog_status = "ready"
            candidate_views = list(ready_task_views)
        records.append(
            {
                "sequence_id": row.sequence_id,
                "split": row.split,
                "frame_count": row.frame_count,
                "n_particles": row.n_particles,
                "particle_group_id": row.particle_group_id,
                "field_status": row.field_status,
                "in_schedule_subset": in_subset,
                "image_shape": image_shape,
                "candidate_task_views": candidate_views,
                "catalog_status": catalog_status,
            }
        )
    return pd.DataFrame(records)


def summarize_task_sample(
    dataset: Any,
    index: int = 0,
    *,
    x_label: str = "x",
    y_label: str = "y",
) -> dict[str, Any]:
    """Compact dict for notebook display of one ``x, y = dataset[i]`` sample."""
    x, y = dataset[index]
    summary: dict[str, Any] = {
        "task": dataset.task.name,
        "selected_row_count": len(dataset),
        "index": index,
        x_label: {},
        y_label: {},
    }
    for key, value in x.items():
        if hasattr(value, "shape"):
            summary[x_label][key] = {"shape": tuple(value.shape), "dtype": str(value.dtype)}
        else:
            summary[x_label][key] = value
    for key, value in y.items():
        if hasattr(value, "shape"):
            summary[y_label][key] = {"shape": tuple(value.shape), "dtype": str(value.dtype)}
        else:
            summary[y_label][key] = value
    return summary
