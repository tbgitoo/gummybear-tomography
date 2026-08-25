"""Shared Milestone 8 Step 3 study helpers for milestone_08 notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tomography_ml.gummybear_data_catalog import estimate_intensity_stats
from tomography_ml.gummybear_data_catalog.task_dataset import (
    DatasetTaskSpec,
    build_task_dataset,
)
from tomography_ml.localization import (
    SingleViewArchConfig,
    build_from_config,
    count_parameters,
    describe_feature_geometry,
    evaluate_split_rmse,
    materialize_lazy_modules,
    train_full_split,
    win3e_architecture_freeze,
    win3e_control_configs,
)
from tomography_ml.studies import M8_CANONICAL_LR_BY_ARCH, set_train_seed
from tomography_ml.training import make_batch_xy_single
from tomography_ml_validation.run_history import effective_n_repeat, next_seed

DEFAULT_N_REPEAT = 5
TRIAD_ROLES = ("primary", "positive_baseline", "negative_control")


def study_results_dir(repo_root: Path | str, subdir: str) -> Path:
    """Directory for one M8 Step 3 study under ``checkpoints/m8/`` (ML artefacts).

    Optical corpora stay under ``data/generated/``; ablation CSVs / freeze JSON
    land beside Final Report checkpoints as ``checkpoints/m8/<subdir>/``.
    """
    return Path(repo_root) / "checkpoints" / "m8" / subdir


def mechanism_task_specs(
    *,
    image_normalize: str = "per_image_minmax",
    optical_setup_id: str = "opt_m8_high_001",
    x_field: str = "anomaly_ref",
) -> tuple[DatasetTaskSpec, DatasetTaskSpec, DatasetTaskSpec]:
    """Train / val / test tasks for Milestone 8 Step 3B–3D mechanism studies.

    Mechanism sweeps (3B–3D) ran before Milestone 8 Step 3G selected per-view z-score;
    keep ``per_image_minmax`` so fresh runs stay comparable to prior 3B–3D
    CSVs when re-run locally.
    """
    common = dict(
        x_fields=(x_field,),
        y_fields=("particle_x", "particle_y", "particle_z"),
        keep_angles_deg=180.0,
        image_normalize=image_normalize,
    )
    train_task = DatasetTaskSpec(
        name="localization_train",
        row_filter={
            "split": "train",
            "field_status": "complete",
            "optical_setup_id": optical_setup_id,
        },
        **common,
    )
    val_task = DatasetTaskSpec(
        name="localization_val",
        row_filter={
            "split": "validation",
            "field_status": "complete",
            "optical_setup_id": optical_setup_id,
        },
        **common,
    )
    test_task = DatasetTaskSpec(
        name="localization_test",
        row_filter={
            "split": "test",
            "field_status": "complete",
            "optical_setup_id": optical_setup_id,
        },
        **common,
    )
    return train_task, val_task, test_task


def capability_task_specs(
    *,
    x_field: str = "anomaly_ref",
    image_normalize: str = "per_image_zscore",
    optical_setup_id: str = "opt_m8_high_001",
    intensity_mean: float | None = None,
    intensity_std: float | None = None,
) -> tuple[DatasetTaskSpec, DatasetTaskSpec, DatasetTaskSpec]:
    """Train / val / test tasks for Milestone 8 Step 3F–3J capability studies."""
    common = dict(
        x_fields=(x_field,),
        y_fields=("particle_x", "particle_y", "particle_z"),
        keep_angles_deg=180.0,
        image_normalize=image_normalize,
        intensity_mean=intensity_mean,
        intensity_std=intensity_std,
    )
    train_task = DatasetTaskSpec(
        name="localization_train",
        row_filter={
            "split": "train",
            "field_status": "complete",
            "optical_setup_id": optical_setup_id,
        },
        **common,
    )
    val_task = DatasetTaskSpec(
        name="localization_val",
        row_filter={
            "split": "validation",
            "field_status": "complete",
            "optical_setup_id": optical_setup_id,
        },
        **common,
    )
    test_task = DatasetTaskSpec(
        name="localization_test",
        row_filter={
            "split": "test",
            "field_status": "complete",
            "optical_setup_id": optical_setup_id,
        },
        **common,
    )
    return train_task, val_task, test_task


def build_task_specs_with_normalisation(
    catalog_rows,
    *,
    x_field: str,
    image_normalize: str,
    optical_setup_id: str = "opt_m8_high_001",
) -> tuple[DatasetTaskSpec, DatasetTaskSpec, DatasetTaskSpec]:
    """Build task specs, estimating train-split stats when required."""
    if image_normalize == "train_split_zscore":
        raw_train, raw_val, raw_test = capability_task_specs(
            x_field=x_field,
            image_normalize="none",
            optical_setup_id=optical_setup_id,
        )
        train_ds_raw = build_task_dataset(catalog_rows, raw_train)
        stats = estimate_intensity_stats(train_ds_raw, x_field)
        return capability_task_specs(
            x_field=x_field,
            image_normalize=image_normalize,
            optical_setup_id=optical_setup_id,
            intensity_mean=stats.mean,
            intensity_std=stats.std,
        )
    return capability_task_specs(
        x_field=x_field,
        image_normalize=image_normalize,
        optical_setup_id=optical_setup_id,
    )


def lr_for_config(cfg: SingleViewArchConfig) -> float:
    return float(M8_CANONICAL_LR_BY_ARCH[cfg.head_type])


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return mu, sd


def aggregate_runs(
    runs_df: pd.DataFrame,
    *,
    group_key: str = "variant",
    sort_metric: str = "validation_RMSE_total",
) -> pd.DataFrame:
    """One summary row per group with mean ± std validation RMSE."""
    if runs_df is None or runs_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    # Axis letters match per_axis_rmse / evaluate_split_rmse (X/Y/Z uppercase).
    metrics = (
        "train_RMSE_total",
        "validation_RMSE_total",
        "test_RMSE_total",
        "best_validation_RMSE_total",
        "train_RMSE_X",
        "train_RMSE_Y",
        "train_RMSE_Z",
        "validation_RMSE_X",
        "validation_RMSE_Y",
        "validation_RMSE_Z",
        "test_RMSE_X",
        "test_RMSE_Y",
        "test_RMSE_Z",
    )
    for group_value, group in runs_df.groupby(group_key, sort=False):
        base = group.iloc[0].to_dict()
        skip = {
            "repeat_index",
            "seed",
            "run_id",
            group_key,
        }
        row = {k: base[k] for k in base if k not in skip}
        for metric in metrics:
            if metric not in group.columns:
                continue
            mu, sd = mean_std(group[metric].astype(float).tolist())
            row[f"{metric}_mean"] = mu
            row[f"{metric}_std"] = sd
        row[group_key] = group_value
        row["n_repeat"] = int(len(group))
        if "seed" in group.columns:
            row["seeds"] = ",".join(str(int(s)) for s in group["seed"].tolist())
        rows.append(row)
    summary = pd.DataFrame(rows)
    mean_col = f"{sort_metric}_mean"
    if mean_col in summary.columns:
        return summary.sort_values(mean_col)
    if sort_metric in summary.columns:
        return summary.sort_values(sort_metric)
    return summary


def load_study_results(
    results_dir: Path | str,
    *,
    summary_name: str,
    runs_name: str,
    group_key: str = "variant",
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load runs CSV (preferred) and/or legacy summary CSV from a study dir."""
    out_dir = Path(results_dir)
    runs_path = out_dir / runs_name
    summary_path = out_dir / summary_name
    runs_df = pd.read_csv(runs_path) if runs_path.is_file() else None
    if runs_df is not None and not runs_df.empty:
        return aggregate_runs(runs_df, group_key=group_key), runs_df
    if summary_path.is_file():
        return pd.read_csv(summary_path), None
    return None, None


def validation_rmse_columns(results_df: pd.DataFrame) -> tuple[str, str | None]:
    if "validation_RMSE_total_mean" in results_df.columns:
        return "validation_RMSE_total_mean", "validation_RMSE_total_std"
    return "validation_RMSE_total", None


def sample_hw(train_ds, train_task: DatasetTaskSpec) -> tuple[torch.Tensor, int, int]:
    x0, _ = train_ds[0]
    views0 = np.asarray(x0[train_task.x_fields[0]])
    sample = views0[0] if views0.ndim == 4 else (views0 if views0.ndim == 3 else views0[None, ...])
    _, height, width = np.asarray(sample).shape
    dummy = torch.as_tensor(sample, dtype=torch.float32).unsqueeze(0)
    return dummy, int(height), int(width)


def triad_role_for_config(cfg: SingleViewArchConfig) -> str:
    freeze = win3e_architecture_freeze()
    if cfg.arch_name == freeze.primary_config().arch_name:
        return "primary"
    if cfg.arch_name == freeze.positive_baseline:
        return "positive_baseline"
    if cfg.arch_name == freeze.negative_control:
        return "negative_control"
    return "other"


def _train_one_variant(
    cfg: SingleViewArchConfig,
    *,
    train_ds,
    val_ds,
    test_ds,
    train_task: DatasetTaskSpec,
    device: torch.device,
    dummy: torch.Tensor,
    height: int,
    width: int,
    num_epochs: int,
    batch_size: int,
    early_stop_patience: int,
    seed: int,
    repeat_index: int,
    run_id: int,
    experiment_prefix: str,
    win: str,
    architecture_factor: str,
    hypothesis: str,
    extra_fields: Mapping[str, Any] | None = None,
    evaluate_test: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_train_seed(int(seed))
    n_outputs = len(train_task.y_fields)
    batch_xy = make_batch_xy_single(
        x_field=train_task.x_fields[0],
        y_fields=train_task.y_fields,
        device=device,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, len(val_ds)),
        shuffle=False,
    )
    test_loader = None
    if evaluate_test and test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=min(batch_size, len(test_ds)),
            shuffle=False,
        )

    model = build_from_config(cfg, n_outputs=n_outputs, device=device)
    materialize_lazy_modules(model, dummy)
    n_params = count_parameters(model)
    geom = describe_feature_geometry(model.encoder, height=height, width=width)
    lr = lr_for_config(cfg)
    fit = train_full_split(
        model=model,
        train_loader=train_loader,
        batch_xy=batch_xy,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        val_loader=val_loader,
        y_fields=train_task.y_fields,
        early_stop_patience=early_stop_patience,
        progress_label=f"{cfg.arch_name} rep {repeat_index + 1}",
    )
    train_metrics = evaluate_split_rmse(
        model=model,
        loader=train_loader,
        batch_xy=batch_xy,
        y_fields=train_task.y_fields,
        prefix="train",
    )
    val_metrics = evaluate_split_rmse(
        model=model,
        loader=val_loader,
        batch_xy=batch_xy,
        y_fields=train_task.y_fields,
        prefix="validation",
    )
    test_metrics: dict[str, Any] = {}
    if test_loader is not None:
        test_metrics = evaluate_split_rmse(
            model=model,
            loader=test_loader,
            batch_xy=batch_xy,
            y_fields=train_task.y_fields,
            prefix="test",
        )
    row = {
        "experiment_id": f"{experiment_prefix}__{cfg.arch_name}",
        "win": win,
        "architecture_factor": architecture_factor,
        "hypothesis": hypothesis,
        "variant": cfg.arch_name,
        "head_type": cfg.head_type,
        "downsample": cfg.downsample,
        "encoder_channels": json.dumps(list(cfg.encoder_channels)),
        "architecture_config": json.dumps(cfg.to_dict()),
        "parameter_count": n_params,
        "feature_map_hw": str(geom["feature_map_hw"]),
        "flatten_length": geom["flatten_length"],
        "lr": lr,
        "input_representation": train_task.x_fields[0],
        "normalisation": train_task.image_normalize,
        "repeat_index": int(repeat_index),
        "seed": int(seed),
        "run_id": int(run_id),
        "best_validation_RMSE_total": fit["best_validation_RMSE_total"],
        **train_metrics,
        **val_metrics,
        **test_metrics,
        "interpretation": "",
    }
    if extra_fields:
        row.update(dict(extra_fields))
    return row, fit["history"]


def run_architecture_grid_study(
    train_ds,
    val_ds,
    train_task: DatasetTaskSpec,
    *,
    device: torch.device | str,
    configs: Sequence[SingleViewArchConfig],
    win: str,
    architecture_factor: str,
    hypothesis: str,
    experiment_prefix: str,
    summary_csv_name: str,
    runs_csv_name: str,
    num_epochs: int = 200,
    batch_size: int = 16,
    early_stop_patience: int = 40,
    results_dir: Path | str | None = None,
    write_csv: bool = True,
    n_repeat: int = DEFAULT_N_REPEAT,
    base_seed: int = 0,
    test_ds=None,
    evaluate_test: bool = False,
    extra_fields: Callable[[SingleViewArchConfig], Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Train each architecture variant ``n_repeat`` times with distinct seeds."""
    configs = tuple(configs)
    device = torch.device(device)
    dummy, height, width = sample_hw(train_ds, train_task)
    dummy = dummy.to(device)

    n_rep = effective_n_repeat(int(n_repeat))
    out_dir = Path(results_dir) if results_dir is not None else None
    runs_csv = out_dir / runs_csv_name if out_dir is not None else None
    session_base_seed = next_seed(runs_csv, default=int(base_seed)) if runs_csv else int(base_seed)

    run_rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    run_id = 1

    for rep in range(n_rep):
        seed = int(session_base_seed) + int(rep)
        if n_rep > 1:
            print(f"\n######## M8 Step {win} repeat {rep + 1}/{n_rep}  seed={seed} ########")
        for cfg in configs:
            fields = extra_fields(cfg) if extra_fields is not None else {}
            row, history = _train_one_variant(
                cfg,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                train_task=train_task,
                device=device,
                dummy=dummy,
                height=height,
                width=width,
                num_epochs=num_epochs,
                batch_size=batch_size,
                early_stop_patience=early_stop_patience,
                seed=seed,
                repeat_index=rep,
                run_id=run_id,
                experiment_prefix=experiment_prefix,
                win=win,
                architecture_factor=architecture_factor,
                hypothesis=hypothesis,
                extra_fields=fields,
                evaluate_test=evaluate_test,
            )
            run_rows.append(row)
            histories[cfg.arch_name] = history
            run_id += 1
            print(
                f"{cfg.arch_name:22s}  seed={seed}  "
                f"train_RMSE={row['train_RMSE_total']:.4f}  "
                f"val_RMSE={row['validation_RMSE_total']:.4f}"
            )

    runs_df = pd.DataFrame(run_rows)
    summary_df = aggregate_runs(runs_df)
    if write_csv and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        runs_df.to_csv(out_dir / runs_csv_name, index=False)
        summary_df.to_csv(out_dir / summary_csv_name, index=False)
    return summary_df, runs_df, histories


def _flush_triad_study_csvs(
    run_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    runs_csv_name: str,
    summary_csv_name: str,
    factor_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs_df = assign_triad_run_group(pd.DataFrame(run_rows), factor_column)
    summary_df = aggregate_runs(runs_df, group_key="run_group")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(out_dir / runs_csv_name, index=False)
    summary_df.to_csv(out_dir / summary_csv_name, index=False)
    return summary_df, runs_df


def run_triad_factor_study(
    catalog_rows,
    *,
    factors: Sequence[Any],
    factor_column: str,
    factor_label: Callable[[Any], str] | None = None,
    build_tasks_for_factor: Callable[[Any], tuple[DatasetTaskSpec, DatasetTaskSpec, DatasetTaskSpec]],
    device: torch.device | str,
    win: str,
    architecture_factor: str,
    hypothesis: str,
    experiment_prefix: str,
    summary_csv_name: str,
    runs_csv_name: str,
    num_epochs: int = 200,
    batch_size: int = 16,
    early_stop_patience: int = 40,
    results_dir: Path | str | None = None,
    write_csv: bool = True,
    n_repeat: int = DEFAULT_N_REPEAT,
    base_seed: int = 0,
    evaluate_test: bool = False,
    resume: bool = True,
    extra_fields_for_factor: Callable[[Any, SingleViewArchConfig], Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Train the Milestone 8 Step 3E control triad for each ladder factor.

    When ``resume=True`` and a runs CSV already exists, completed
    ``(factor, variant, seed)`` rows are skipped and new rows are appended.
    CSVs are flushed after each finished train so interrupted long studies
    keep partial progress.
    """
    label_fn = factor_label or (lambda factor: str(getattr(factor, "name")))
    configs = win3e_control_configs()
    device = torch.device(device)
    n_rep = effective_n_repeat(int(n_repeat))
    out_dir = Path(results_dir) if results_dir is not None else None
    runs_csv = out_dir / runs_csv_name if out_dir is not None else None

    run_rows: list[dict[str, Any]] = []
    done: set[tuple[str, str, int]] = set()
    if resume and runs_csv is not None and runs_csv.is_file():
        prior = pd.read_csv(runs_csv)
        if not prior.empty:
            run_rows = prior.to_dict(orient="records")
            if factor_column in prior.columns and "variant" in prior.columns and "seed" in prior.columns:
                done = {
                    (str(r[factor_column]), str(r["variant"]), int(r["seed"]))
                    for r in run_rows
                }
            print(f"Resuming with {len(run_rows)} existing runs ({len(done)} keys).")

    session_base_seed = int(base_seed)
    histories: dict[str, list[dict[str, Any]]] = {}
    run_id = len(run_rows) + 1

    for factor in factors:
        factor_name = label_fn(factor)
        train_task, val_task, test_task = build_tasks_for_factor(factor)
        train_ds = build_task_dataset(catalog_rows, train_task)
        val_ds = build_task_dataset(catalog_rows, val_task)
        test_ds = build_task_dataset(catalog_rows, test_task) if evaluate_test else None
        dummy, height, width = sample_hw(train_ds, train_task)
        dummy = dummy.to(device)

        for rep in range(n_rep):
            seed = int(session_base_seed) + int(rep)
            if n_rep > 1:
                print(
                    f"\n######## M8 Step {win} {factor_column}={factor_name} "
                    f"repeat {rep + 1}/{n_rep}  seed={seed} ########"
                )
            for cfg in configs:
                key_done = (factor_name, cfg.arch_name, int(seed))
                if key_done in done:
                    print(
                        f"{factor_name:12s} {cfg.arch_name:22s}  seed={seed}  "
                        f"(skip — already in runs CSV)"
                    )
                    continue
                extra = {
                    factor_column: factor_name,
                    "triad_role": triad_role_for_config(cfg),
                }
                if extra_fields_for_factor is not None:
                    extra.update(extra_fields_for_factor(factor, cfg))
                row, history = _train_one_variant(
                    cfg,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    test_ds=test_ds,
                    train_task=train_task,
                    device=device,
                    dummy=dummy,
                    height=height,
                    width=width,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    early_stop_patience=early_stop_patience,
                    seed=seed,
                    repeat_index=rep,
                    run_id=run_id,
                    experiment_prefix=experiment_prefix,
                    win=win,
                    architecture_factor=architecture_factor,
                    hypothesis=hypothesis,
                    extra_fields=extra,
                    evaluate_test=evaluate_test,
                )
                run_rows.append(row)
                done.add(key_done)
                key = f"{factor_name}::{cfg.arch_name}"
                histories[key] = history
                run_id += 1
                print(
                    f"{factor_name:12s} {cfg.arch_name:22s}  seed={seed}  "
                    f"val_RMSE={row['validation_RMSE_total']:.4f}"
                )
                if write_csv and out_dir is not None:
                    _flush_triad_study_csvs(
                        run_rows,
                        out_dir=out_dir,
                        runs_csv_name=runs_csv_name,
                        summary_csv_name=summary_csv_name,
                        factor_column=factor_column,
                    )

    if not run_rows:
        return pd.DataFrame(), pd.DataFrame(), histories

    if write_csv and out_dir is not None:
        summary_df, runs_df = _flush_triad_study_csvs(
            run_rows,
            out_dir=out_dir,
            runs_csv_name=runs_csv_name,
            summary_csv_name=summary_csv_name,
            factor_column=factor_column,
        )
    else:
        runs_df = assign_triad_run_group(pd.DataFrame(run_rows), factor_column)
        summary_df = aggregate_runs(runs_df, group_key="run_group")
    return summary_df, runs_df, histories


def assign_triad_run_group(runs_df: pd.DataFrame, factor_key: str) -> pd.DataFrame:
    """Add ``run_group`` = ``{factor}::{variant}`` for triad aggregation."""
    out = runs_df.copy()
    out["run_group"] = out[factor_key].astype(str) + "::" + out["variant"].astype(str)
    return out


def write_json_artifact(path: Path | str, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
