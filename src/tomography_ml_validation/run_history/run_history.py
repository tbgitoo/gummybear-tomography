"""Append-only experiment run history for localisation notebooks.

Harmonized CSV table schema shared by 10_1A / 10_1B (and reusable elsewhere).
Lives under ``tomography_ml_validation`` (bookkeeping / validation companion),
not the scientific ``tomography_ml`` runtime.

Each completed train+eval writes one row per ``variant_id`` with an
incremented ``run_id``. Aggregate with :func:`aggregate_run_history` for
mean ± std root-mean-square error (RMSE) plots.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# Bump when column semantics change (append new optional columns freely).
# Integer schema version written on each appended run-history row.
RUN_HISTORY_SCHEMA_VERSION = 1

# Default basename for append-only experiment CSV table logs.
DEFAULT_HISTORY_FILENAME = "run_history.csv"

# Default basename for pre-aggregated per-session summary tables.
DEFAULT_SESSION_SUMMARY_FILENAME = "session_summary.csv"

# Core columns written for every variant row (extra keys are preserved on append).
HISTORY_CORE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "repeat_index",
    "timestamp_utc",
    "schema_version",
    "notebook_id",
    "experiment_id",
    "variant_id",
    "fusion_pattern",
    "backbone_kind",
    "architecture_note",
    "architecture_json",
    "freeze_encoder",
    "end_to_end",
    "learned_parameter_count",
    "n_views",
    "n_lights",
    "fixed_camera_deg",
    "single_light_ref_deg",
    "light_geometry",
    "lr_stage_a",
    "lr_stage_b",
    "epochs_max",
    "early_stop_patience",
    "epochs_ran_stage_a",
    "epochs_ran_stage_b",
    "batch_size",
    "quick",
    "run_lr_study",
    "include_pooled_control",
    "n_repeat_training_requested",
    "n_repeat_training_effective",
    "seed",
    "RMSE_train_total",
    "RMSE_train_X",
    "RMSE_train_Y",
    "RMSE_train_Z",
    "RMSE_validation_total",
    "RMSE_validation_X",
    "RMSE_validation_Y",
    "RMSE_validation_Z",
    "RMSE_test_total",
    "RMSE_test_X",
    "RMSE_test_Y",
    "RMSE_test_Z",
)

# Ordered core CSV table columns; additional keys are appended after these on write.


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string for ``timestamp_utc``."""
    return datetime.now(timezone.utc).isoformat()


def effective_n_repeat(
    n_repeat_training: int,
    *,
    run_lr_study: bool = False,
) -> int:
    """Resolve how many full train+eval repeats to run.

    ``RUN_LR_STUDY`` forces a single repeat so LR sweeps are not multiplied.
    """
    n = int(n_repeat_training)
    if n < 1:
        n = 1
    if run_lr_study and n > 1:
        return 1
    return n


def next_run_id(path: Path | str) -> int:
    """Return the next integer ``run_id`` for an append-only history CSV table.

    Starts at ``1`` when the file is missing or empty.

    Args:
        path: On-disk ``run_history.csv`` path.

    Returns:
        int: ``max(run_id) + 1`` from existing rows.
    """
    path = Path(path)
    if not path.is_file():
        return 1
    prev = pd.read_csv(path)
    if prev.empty or "run_id" not in prev.columns:
        return 1
    return int(prev["run_id"].max()) + 1


def next_seed(path: Path | str, *, default: int = 0) -> int:
    """Next integer training seed continuing an on-disk history CSV table.

    Uses ``max(numeric seed) + 1`` when history has parseable seeds; otherwise
    returns ``default`` (typically the notebook ``BASE_SEED``). Empty cells and
    non-numeric seeds are ignored. Mirrors :func:`next_run_id` so a notebook
    restart does not reuse seeds ``0..N-1`` after prior runs were appended.
    """
    path = Path(path)
    default_i = int(default)
    if not path.is_file():
        return default_i
    prev = pd.read_csv(path)
    if prev.empty or "seed" not in prev.columns:
        return default_i
    values = pd.to_numeric(prev["seed"], errors="coerce").dropna()
    if values.empty:
        return default_i
    return int(values.max()) + 1


def append_run_history(
    path: Path | str,
    new_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Append harmonized rows to a run-history CSV table (create if missing).

    Ensures ``schema_version`` is set on new rows. Column order is stable:
    :data:`HISTORY_CORE_COLUMNS` first, then any extra keys.

    Args:
        path: Destination CSV path (parent directories are created).
        new_rows: One mapping per variant/run to append.

    Returns:
        pandas.DataFrame: Full table after append (old + new rows).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(list(new_rows))
    if "schema_version" not in new_df.columns:
        new_df["schema_version"] = RUN_HISTORY_SCHEMA_VERSION
    if path.is_file():
        old = pd.read_csv(path)
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    # Stable column order: known cores first, then extras.
    ordered = [c for c in HISTORY_CORE_COLUMNS if c in out.columns]
    ordered.extend(c for c in out.columns if c not in ordered)
    out = out[ordered]
    out.to_csv(path, index=False)
    return out


def _metric_rmse(metrics: Mapping[str, Any] | None, axis: str) -> float:
    if metrics is None:
        return float("nan")
    key = "train_RMSE_total" if axis == "total" else f"train_RMSE_{axis}"
    try:
        return float(metrics[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def build_history_row(
    *,
    run_id: int,
    repeat_index: int,
    notebook_id: str,
    experiment_id: str,
    variant_id: str,
    fusion_pattern: str,
    backbone_kind: str,
    metrics_by_split: Mapping[str, Mapping[str, Any]],
    architecture_note: str = "",
    architecture_json: str = "",
    freeze_encoder: bool | None = None,
    end_to_end: bool | None = None,
    learned_parameter_count: float | int | None = None,
    n_views: int = 0,
    n_lights: int | None = None,
    fixed_camera_deg: float = float("nan"),
    single_light_ref_deg: float = float("nan"),
    light_geometry: bool = False,
    lr_stage_a: float = float("nan"),
    lr_stage_b: float = float("nan"),
    epochs_max: int = 0,
    early_stop_patience: int = 0,
    epochs_ran_stage_a: float = float("nan"),
    epochs_ran_stage_b: float = float("nan"),
    batch_size: int = 0,
    quick: bool = False,
    run_lr_study: bool = False,
    include_pooled_control: bool = False,
    n_repeat_training_requested: int = 1,
    n_repeat_training_effective: int = 1,
    seed: str | int | None = "",
    timestamp_utc: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one harmonized CSV row after a variant finishes train+eval.

    Populates root-mean-square error (RMSE) columns from ``metrics_by_split`` using keys like
    ``train_RMSE_total`` and ``train_RMSE_X``. Unknown keys in ``extra`` are
    appended when not already present.

    Args:
        run_id: Monotonic run identifier shared across variants in one session.
        repeat_index: Zero-based repeat index within the requested repeat count.
        notebook_id: Notebook slug (for example ``"10_1A"``).
        experiment_id: Logical experiment name within the notebook.
        variant_id: Stable variant key used in plots.
        fusion_pattern: Fusion architecture label string.
        backbone_kind: ``"fourier"`` or ``"pooled"`` (plot grouping).
        metrics_by_split: Nested dict ``split → {train_RMSE_*}`` metrics.
        architecture_note, architecture_json: Free-text architecture metadata.
        freeze_encoder, end_to_end: Training mode flags.
        learned_parameter_count: Trainable parameter count when known.
        n_views: Number of camera views in the task.
        n_lights: Number of illumination conditions (defaults to ``n_views``).
        fixed_camera_deg, single_light_ref_deg: Geometry reference angles.
        light_geometry: Whether light-direction features are enabled.
        lr_stage_a, lr_stage_b: Stage learning rates.
        epochs_max, early_stop_patience: Training schedule caps.
        epochs_ran_stage_a, epochs_ran_stage_b: Actual epochs executed.
        batch_size: Minibatch size.
        quick: Quick-run flag for smoke configs.
        run_lr_study: Whether an LR sweep suppressed multi-repeat training.
        include_pooled_control: Whether pooled variants were requested.
        n_repeat_training_requested, n_repeat_training_effective: Repeat counts.
        seed: Training seed stored as string in CSV.
        timestamp_utc: Optional override; defaults to :func:`utc_now_iso`.
        extra: Additional columns merged when not colliding with core fields.

    Returns:
        dict: One row ready for :func:`append_run_history`.

    Typically used after :func:`~tomography_ml.training.training_helpers.train_e2e` or
    single-view training; consumed by :func:`~tomography_ml_validation.plotting.illumination_fusion.plot_illumination_fusion_results`.
    """
    row: dict[str, Any] = {
        "run_id": int(run_id),
        "repeat_index": int(repeat_index),
        "timestamp_utc": timestamp_utc or utc_now_iso(),
        "schema_version": RUN_HISTORY_SCHEMA_VERSION,
        "notebook_id": str(notebook_id),
        "experiment_id": str(experiment_id),
        "variant_id": str(variant_id),
        "fusion_pattern": str(fusion_pattern),
        "backbone_kind": str(backbone_kind),
        "architecture_note": str(architecture_note),
        "architecture_json": str(architecture_json),
        "freeze_encoder": freeze_encoder,
        "end_to_end": end_to_end,
        "learned_parameter_count": (
            float("nan")
            if learned_parameter_count is None
            else float(learned_parameter_count)
        ),
        "n_views": int(n_views),
        "n_lights": int(n_views if n_lights is None else n_lights),
        "fixed_camera_deg": float(fixed_camera_deg),
        "single_light_ref_deg": float(single_light_ref_deg),
        "light_geometry": bool(light_geometry),
        "lr_stage_a": float(lr_stage_a),
        "lr_stage_b": float(lr_stage_b),
        "epochs_max": int(epochs_max),
        "early_stop_patience": int(early_stop_patience),
        "epochs_ran_stage_a": float(epochs_ran_stage_a),
        "epochs_ran_stage_b": float(epochs_ran_stage_b),
        "batch_size": int(batch_size),
        "quick": bool(quick),
        "run_lr_study": bool(run_lr_study),
        "include_pooled_control": bool(include_pooled_control),
        "n_repeat_training_requested": int(n_repeat_training_requested),
        "n_repeat_training_effective": int(n_repeat_training_effective),
        "seed": "" if seed is None else str(seed),
    }
    for split in ("train", "validation", "test"):
        m = metrics_by_split.get(split)
        for axis in ("total", "X", "Y", "Z"):
            row[f"RMSE_{split}_{axis}"] = _metric_rmse(m, axis)
    if extra:
        for k, v in extra.items():
            if k not in row:
                row[k] = v
    return row


def _mean_std(vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return mu, sd


def aggregate_run_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Aggregate history rows to one row per ``variant_id`` with mean/std root-mean-square error (RMSE).

    Adds ``RMSE_{split}_{axis}_mean/std`` columns and back-compat aliases
    ``RMSE_{split}_mean/std`` equal to the total axis.

    Args:
        hist: Raw run-history dataframe (multiple rows per variant allowed).

    Returns:
        pandas.DataFrame: Empty when input is empty; otherwise one row per variant
        with ``n_runs`` and comma-separated ``run_ids``.
    """
    if hist is None or len(hist) == 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variant_id, g in hist.groupby("variant_id", sort=False):
        row: dict[str, Any] = {
            "variant_id": variant_id,
            "backbone_kind": g["backbone_kind"].iloc[0]
            if "backbone_kind" in g.columns
            else "",
            "fusion_pattern": g["fusion_pattern"].iloc[0]
            if "fusion_pattern" in g.columns
            else "",
            "n_runs": int(len(g)),
            "run_ids": ",".join(str(int(x)) for x in g["run_id"].tolist())
            if "run_id" in g.columns
            else "",
        }
        if "notebook_id" in g.columns:
            row["notebook_id"] = g["notebook_id"].iloc[0]
        if "experiment_id" in g.columns:
            row["experiment_id"] = g["experiment_id"].iloc[0]
        for split_key in ("train", "validation", "test"):
            for axis in ("total", "X", "Y", "Z"):
                col = f"RMSE_{split_key}_{axis}"
                if col not in g.columns:
                    continue
                mu, sd = _mean_std(g[col].to_numpy(dtype=float))
                row[f"RMSE_{split_key}_{axis}_mean"] = mu
                row[f"RMSE_{split_key}_{axis}_std"] = sd
            if f"RMSE_{split_key}_total_mean" in row:
                row[f"RMSE_{split_key}_mean"] = row[
                    f"RMSE_{split_key}_total_mean"
                ]
                row[f"RMSE_{split_key}_std"] = row[f"RMSE_{split_key}_total_std"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_summary_for_plots(
    history_path: Path | str,
    *,
    session_run_ids: Sequence[int] | None = None,
    session_summary: pd.DataFrame | None = None,
    session_summary_path: Path | str | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """Resolve a summary table for mean±std root-mean-square error (RMSE) plotting.

    Preference order: filter on-disk history to ``session_run_ids`` → full
    history → in-memory ``session_summary`` → load ``session_summary_path``.

    Args:
        history_path: Append-only run history CSV table.
        session_run_ids: Optional ``run_id`` filter before aggregation.
        session_summary: Pre-aggregated summary dataframe.
        session_summary_path: On-disk session summary CSV table fallback.

    Returns:
        tuple: ``(summary_df, summary_src)`` where ``summary_df`` may be None
        when no source is available; ``summary_src`` explains which path was used.
    """
    history_path = Path(history_path)
    if history_path.is_file():
        hist_all = pd.read_csv(history_path)
        if len(hist_all):
            if session_run_ids:
                hist_use = hist_all[hist_all["run_id"].isin(list(session_run_ids))]
                if len(hist_use) == 0:
                    hist_use = hist_all
                src = f"history run_ids={list(session_run_ids)} (n_rows={len(hist_use)})"
            else:
                hist_use = hist_all
                src = f"full history (n_rows={len(hist_use)})"
            return aggregate_run_history(hist_use), src
    if session_summary is not None and len(session_summary):
        return session_summary.copy(), "in-memory session_summary"
    if session_summary_path is not None:
        p = Path(session_summary_path)
        if p.is_file():
            return pd.read_csv(p), f"loaded {p}"
    return None, "no history / session summary"
