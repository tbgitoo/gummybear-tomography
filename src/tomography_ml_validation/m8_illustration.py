"""Thin helpers so didactic notebooks can load an M8 sample in one line."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from gummybear.paths import repo_relative_path
from tomography_ml.gummybear_data_catalog import build_catalog_rows, load_catalog_jobs
from tomography_ml.gummybear_data_catalog.task_dataset import (
    CatalogTaskDataset,
    DatasetTaskSpec,
    build_task_dataset,
)

_M8_FULL_WORKBOOK = Path("configs") / "m8" / "localization_single_particle.xlsx"
_M8_DEMO_WORKBOOK = Path("configs") / "m8" / "m8_demo.xlsx"
_M8_FULL_OUTPUT = Path("data") / "generated" / "m8_1" / "single_particle"
_M8_DEMO_OUTPUT = Path("data") / "generated" / "m8_demo"


def default_m8_illustration_task(*, keep_angles_deg: float = 0.0) -> DatasetTaskSpec:
    """Canonical single-view M8 localisation task used by the CNN walkthrough."""
    return DatasetTaskSpec(
        name="localization_M8",
        row_filter={"split": "train", "field_status": "complete"},
        x_fields=("anomaly_ref",),
        y_fields=("particle_x", "particle_y", "particle_z"),
        keep_angles_deg=keep_angles_deg,
    )


def resolve_m8_illustration_paths(
    repo_root: Path,
    data_mode: str,
    *,
    workbook_path: Path | str | None = None,
    output_root: Path | str | None = None,
) -> tuple[Path, Path]:
    """Return ``(workbook, output_root)`` for the M8 corpus matching ``data_mode``."""
    root = Path(repo_root)
    mode = str(data_mode).strip().lower()
    if mode == "full":
        workbook = root / _M8_FULL_WORKBOOK
        out = root / _M8_FULL_OUTPUT
    else:
        # inspect / demo share the demo corpus in ML illustration cells
        workbook = root / _M8_DEMO_WORKBOOK
        out = root / _M8_DEMO_OUTPUT

    if workbook_path is not None and "m8" in Path(workbook_path).as_posix():
        workbook = Path(workbook_path)
    if output_root is not None and "m8" in Path(output_root).as_posix():
        out = Path(output_root)
    return workbook, out


def ensure_m8_illustration_dataset(
    repo_root: Path | str,
    data_mode: str,
    namespace: Mapping[str, Any] | None = None,
    *,
    workbook_path: Path | str | None = None,
    output_root: Path | str | None = None,
    dataset: CatalogTaskDataset | None = None,
    task: DatasetTaskSpec | None = None,
    keep_angles_deg: float = 0.0,
) -> tuple[CatalogTaskDataset, DatasetTaskSpec]:
    """Return ``(dataset, task)`` for the hard-coded M8 CNN illustration.

    Reuses ``dataset`` / ``task`` when already present (e.g. earlier notebook
    cells). Otherwise loads quietly from the on-disk M8 corpus produced by the
    optics pipeline. Pass ``globals()`` as ``namespace`` to pick up live
    ``WORKBOOK_PATH`` / ``OUTPUT_ROOT`` / existing dataset bindings.
    """
    ns = dict(namespace or {})
    if dataset is None:
        dataset = ns.get("dataset_M8")
    if task is None:
        task = ns.get("localization_task_M8")
    if workbook_path is None:
        workbook_path = ns.get("WORKBOOK_PATH")
    if output_root is None:
        output_root = ns.get("OUTPUT_ROOT")

    if dataset is not None and task is not None:
        return dataset, task

    task = task or default_m8_illustration_task(keep_angles_deg=keep_angles_deg)
    workbook, out = resolve_m8_illustration_paths(
        Path(repo_root),
        data_mode,
        workbook_path=workbook_path,
        output_root=output_root,
    )
    jobs = load_catalog_jobs(workbook, root_path=Path(repo_root), stl_root=Path(repo_root))
    jobs = [replace(job, output_root=str(repo_relative_path(out))) for job in jobs]
    dataset = build_task_dataset(build_catalog_rows(jobs), task)
    return dataset, task
