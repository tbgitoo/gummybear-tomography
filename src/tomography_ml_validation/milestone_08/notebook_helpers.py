"""Notebook-facing helpers for Milestone 8 single-view localisation demos."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from gummybear.paths import display_path, repo_relative_path
from tomography_ml.gummybear_data_catalog import (
    build_catalog_rows,
    load_catalog_jobs,
)
from tomography_ml.gummybear_data_catalog.catalog import CatalogRow
from tomography_ml.gummybear_data_catalog.task_dataset import (
    DatasetTaskSpec,
    build_task_dataset,
)
from tomography_ml.localization import (
    SingleViewArchConfig,
    default_mechanism_grid,
    win3b_receptive_field_grid,
    win3c_channel_capacity_grid,
    win3d_head_expressiveness_grid,
    win3e_architecture_freeze,
    win3e_control_configs,
    win3f_representation_grid,
    win3f_selected_representation,
    win3g_normalisation_grid,
    win3g_selected_normalisation,
    win3h_optical_regime_grid,
    win3i_key_result_sources,
    win3j_single_view_freeze,
)
from tomography_ml_validation.m8_illustration import resolve_m8_illustration_paths

FINAL_REPORT_REL = "GummyBearTomography_Final_Report.ipynb"
M8_PLAN_REL = "plans/milestone_08/08_single_view_localization_plan.md"

# Friendly labels for optical setups used in panels / tables.
REGIME_LABEL_BY_SETUP = {
    "opt_m8_low_001": "low",
    "opt_m8_med_001": "medium",
    "opt_m8_high_001": "high",
}


def m8_corpus_paths(
    repo_root: Path | str,
    *,
    data_mode: str = "full",
) -> dict[str, Path]:
    """Resolve workbook / output roots for demo or full M8 corpora."""
    root = Path(repo_root)
    workbook, output_root = resolve_m8_illustration_paths(root, data_mode)
    return {
        "repo_root": root,
        "workbook_path": workbook,
        "output_root": output_root,
        "cache_root": output_root / "_cache",
        "results_root": root / "checkpoints" / "m8",
    }


def load_m8_catalog_rows(
    repo_root: Path | str,
    *,
    data_mode: str = "full",
) -> tuple[CatalogRow, ...]:
    """Load flat catalog rows for the chosen M8 corpus."""
    paths = m8_corpus_paths(repo_root, data_mode=data_mode)
    jobs = load_catalog_jobs(
        paths["workbook_path"],
        root_path=paths["repo_root"],
        stl_root=paths["repo_root"],
    )
    # Anchor outputs to the resolved scenario folder (workbook may store relative).
    from dataclasses import replace

    jobs = [
        replace(job, output_root=str(repo_relative_path(paths["output_root"])))
        for job in jobs
    ]
    return build_catalog_rows(jobs)


def _regime_label_for_row(row: CatalogRow) -> str | None:
    """Map a catalog row to ``low`` / ``medium`` / ``high`` when possible."""
    setup = getattr(row, "optical_setup_id", None) or ""
    label = REGIME_LABEL_BY_SETUP.get(str(setup))
    if label is not None:
        return label
    sid = row.sequence_id.lower()
    if "low" in sid:
        return "low"
    if "med" in sid:
        return "medium"
    if "high" in sid:
        return "high"
    return None


def _particle_index_token(sequence_id: str) -> str | None:
    """Trailing numeric token for matched multi-regime families (e.g. ``000001``)."""
    parts = str(sequence_id).rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return parts[1]


def pick_regime_exemplars(
    rows: Sequence[CatalogRow],
    *,
    prefer_complete: bool = True,
    particle_index: str | int | None = "000001",
    sequence_ids: Mapping[str, str] | None = None,
) -> dict[str, CatalogRow]:
    """One representative row per optical regime (low / medium / high).

    Prefer a **matched particle identity** across regimes so the Δ hotspot
    location is comparable (same centre, different background optics only).

    Selection order:

    1. Explicit ``sequence_ids`` map ``{low|medium|high: sequence_id}`` when given.
    2. Else matched ``particle_index`` (default ``000001``) across all three regimes.
    3. Else first complete row per regime in catalog order (legacy fallback).

    Returns regimes in fixed display order: ``low``, ``medium``, ``high``.
    """
    by_id = {row.sequence_id: row for row in rows}
    ordered: dict[str, CatalogRow] = {}

    if sequence_ids is not None:
        for label in ("low", "medium", "high"):
            sid = sequence_ids.get(label)
            if sid is None:
                continue
            row = by_id.get(str(sid))
            if row is None:
                raise KeyError(f"sequence_id {sid!r} not in catalog for regime {label!r}")
            if prefer_complete and row.field_status != "complete":
                raise ValueError(
                    f"Pinned exemplar {sid!r} has field_status={row.field_status!r}"
                )
            ordered[label] = row
        if len(ordered) == 3:
            return ordered

    if particle_index is not None:
        token = f"{int(particle_index):06d}" if str(particle_index).isdigit() else str(
            particle_index
        )
        matched: dict[str, CatalogRow] = {}
        for row in rows:
            if _particle_index_token(row.sequence_id) != token:
                continue
            label = _regime_label_for_row(row)
            if label is None:
                continue
            if prefer_complete and row.field_status != "complete":
                continue
            matched.setdefault(label, row)
        if set(matched) == {"low", "medium", "high"}:
            return {label: matched[label] for label in ("low", "medium", "high")}

    # Legacy fallback: first complete row per regime (may mix particle identities).
    by_regime: dict[str, CatalogRow] = {}
    for row in rows:
        label = _regime_label_for_row(row)
        if label is None:
            continue
        if prefer_complete and row.field_status != "complete":
            continue
        by_regime.setdefault(label, row)
        if len(by_regime) == 3:
            break
    return {label: by_regime[label] for label in ("low", "medium", "high") if label in by_regime}


def default_single_view_task(
    *,
    x_field: str = "anomaly_ref",
    keep_angles_deg: float = 180.0,
    optical_setup_id: str | None = "opt_m8_high_001",
    split: str = "train",
) -> DatasetTaskSpec:
    """Canonical M8 single-view localisation task (matches freeze / report)."""
    row_filter: dict[str, Any] = {
        "split": split,
        "field_status": "complete",
    }
    if optical_setup_id is not None:
        row_filter["optical_setup_id"] = optical_setup_id
    return DatasetTaskSpec(
        name="localization_M8",
        row_filter=row_filter,
        x_fields=(x_field,),
        y_fields=("particle_x", "particle_y", "particle_z"),
        keep_angles_deg=keep_angles_deg,
        image_normalize="per_image_zscore",
    )


def build_m8_localization_dataset(
    repo_root: Path | str,
    *,
    data_mode: str = "full",
    task: DatasetTaskSpec | None = None,
):
    """Catalog → task dataset for single-view localisation demos."""
    rows = load_m8_catalog_rows(repo_root, data_mode=data_mode)
    task = task or default_single_view_task()
    return build_task_dataset(rows, task), task


def records_dataframe(records: Sequence[Any]) -> pd.DataFrame:
    """Flatten dataclass / mapping records for notebook display."""
    rows: list[dict[str, Any]] = []
    for item in records:
        if is_dataclass(item) and not isinstance(item, type):
            rows.append(asdict(item))
        elif isinstance(item, Mapping):
            rows.append(dict(item))
        elif hasattr(item, "to_dict"):
            rows.append(dict(item.to_dict()))
        else:
            rows.append({"value": item})
    return pd.DataFrame(rows)


def freeze_record_dataframe(record: Any) -> pd.DataFrame:
    """One-row DataFrame for a freeze dataclass (nested dicts as strings)."""
    if not is_dataclass(record) or isinstance(record, type):
        raise TypeError("freeze_record_dataframe expects a dataclass instance")
    payload = asdict(record)
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple)):
            flat[key] = value if not isinstance(value, dict) else str(value)
        else:
            flat[key] = value
    return pd.DataFrame([flat])


def architecture_grid_dataframe(
    configs: Sequence[SingleViewArchConfig] | None = None,
    *,
    image_hw: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Display table for an M8 Step 3 architecture grid.

    When ``image_hw=(H, W)`` is given, adds explicit MaxPool schedule columns
    (``maxpool_after_blocks``, ``pool_schedule``, ``spatial_hw_path``,
    ``feature_map_hw``) so downsampling presets are not opaque labels.
    """
    from tomography_ml.localization.encoder import describe_downsample_schedule

    configs = configs if configs is not None else default_mechanism_grid()
    height, width = (None, None) if image_hw is None else (int(image_hw[0]), int(image_hw[1]))
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        row = asdict(cfg) if is_dataclass(cfg) and not isinstance(cfg, type) else dict(cfg.to_dict())
        sched = describe_downsample_schedule(
            len(cfg.encoder_channels),
            cfg.downsample,
            height=height,
            width=width,
        )
        row["maxpool_after_blocks"] = sched["maxpool_after_blocks"]
        row["pool_schedule"] = sched["pool_schedule"]
        if sched["spatial_hw_path"] is not None:
            row["spatial_hw_path"] = sched["spatial_hw_path"]
            h_out, w_out = sched["feature_map_hw"]
            row["feature_map_hw"] = f"{h_out}×{w_out}"
        rows.append(row)
    df = pd.DataFrame(rows)
    preferred = [
        "arch_name",
        "head_type",
        "encoder_channels",
        "downsample",
        "maxpool_after_blocks",
        "pool_schedule",
        "spatial_hw_path",
        "feature_map_hw",
        "pre_flatten_channels",
        "embed_dim",
        "flatten_hidden",
        "flatten_head",
        "input_representation",
        "normalisation",
    ]
    ordered = [col for col in preferred if col in df.columns]
    rest = [col for col in df.columns if col not in ordered]
    return df[ordered + rest]


def win3_grids_summary(*, image_hw: tuple[int, int] | None = None) -> dict[str, pd.DataFrame]:
    """All predefined Milestone 8 Step 3B–3H grids as display tables."""
    return {
        "3B_receptive_field": architecture_grid_dataframe(
            win3b_receptive_field_grid(),
            image_hw=image_hw,
        ),
        "3C_channel_capacity": architecture_grid_dataframe(win3c_channel_capacity_grid()),
        "3D_head_expressiveness": architecture_grid_dataframe(
            win3d_head_expressiveness_grid()
        ),
        "3E_controls": architecture_grid_dataframe(win3e_control_configs()),
        "3F_representation": records_dataframe(win3f_representation_grid()),
        "3G_normalisation": records_dataframe(win3g_normalisation_grid()),
        "3H_optical_regime": records_dataframe(win3h_optical_regime_grid()),
        "3I_sources": records_dataframe(win3i_key_result_sources()),
    }


def assert_win3j_freeze_contract() -> dict[str, Any]:
    """Assert inscribed Milestone 8 Step 3J / M8 freeze fields match the plan Conclusion."""
    block = win3j_single_view_freeze()
    arch = win3e_architecture_freeze()
    rep = win3f_selected_representation()
    norm = win3g_selected_normalisation()

    assert arch.selected_variant == "fourier_base_mlp"
    assert arch.spatial_readout_type == "fourier_coded_pool"
    assert arch.widths == (16, 32, 64)
    assert arch.downsampling == "base"
    assert arch.head == "mlp"
    assert arch.library_class == "LocalizerSingleViewFourier"

    assert block.architecture.selected_variant == arch.selected_variant
    assert block.x_field == rep.x_field == "anomaly_ref"
    assert block.image_normalize == norm.image_normalize == "per_image_zscore"
    assert block.keep_angles_deg == 180.0
    assert block.optical_setup_id_reference == "opt_m8_high_001"
    assert block.lr_primary == 0.03
    assert block.library_class == "LocalizerSingleViewFourier"

    return {
        "selected_variant": arch.selected_variant,
        "x_field": block.x_field,
        "image_normalize": block.image_normalize,
        "keep_angles_deg": block.keep_angles_deg,
        "optical_setup_id_reference": block.optical_setup_id_reference,
        "library_class": block.library_class,
        "n_freeze_fields": len(block.freeze_fields),
    }


def describe_paths(paths: Mapping[str, Path]) -> str:
    """Repo-relative path summary for notebook stdout."""
    bits = []
    for key in ("workbook_path", "output_root", "cache_root"):
        if key in paths:
            bits.append(f"{key}={display_path(paths[key])}")
    return "  ".join(bits)


def load_historical_win3_csv(
    repo_root: Path | str,
    relative_csv: str,
    *,
    scenario_root: Path | str | None = None,
) -> pd.DataFrame | None:
    """Load an M8 Step 3 study CSV under ``checkpoints/m8/`` (or an explicit root)."""
    root = Path(repo_root)
    base = Path(scenario_root) if scenario_root is not None else (
        root / "checkpoints" / "m8"
    )
    candidates = [
        base / relative_csv,
        root / "checkpoints" / "m8" / relative_csv,
        # Legacy mistaken location (pre-move): data/generated/m8_1/_win3*/
        root / "data" / "generated" / "m8_1" / relative_csv,
    ]
    for path in candidates:
        if path.is_file():
            return pd.read_csv(path)
    return None
