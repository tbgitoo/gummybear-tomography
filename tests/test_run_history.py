"""Tests for harmonized localisation run-history helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tomography_ml_validation.run_history import (
    RUN_HISTORY_SCHEMA_VERSION,
    aggregate_run_history,
    append_run_history,
    build_history_row,
    effective_n_repeat,
    load_summary_for_plots,
    next_run_id,
    next_seed,
)


def test_effective_n_repeat_and_lr_study_force() -> None:
    assert effective_n_repeat(3) == 3
    assert effective_n_repeat(0) == 1
    assert effective_n_repeat(3, run_lr_study=True) == 1
    assert effective_n_repeat(1, run_lr_study=True) == 1


def test_next_run_id_and_append(tmp_path: Path) -> None:
    path = tmp_path / "hist.csv"
    assert next_run_id(path) == 1
    row = build_history_row(
        run_id=1,
        repeat_index=0,
        notebook_id="10_1A",
        experiment_id="exp",
        variant_id="v1",
        fusion_pattern="p",
        backbone_kind="fourier",
        metrics_by_split={
            "train": {
                "train_RMSE_total": 1.0,
                "train_RMSE_X": 1.1,
                "train_RMSE_Y": 1.2,
                "train_RMSE_Z": 1.3,
            },
            "validation": {
                "train_RMSE_total": 2.0,
                "train_RMSE_X": 2.1,
                "train_RMSE_Y": 2.2,
                "train_RMSE_Z": 2.3,
            },
            "test": {
                "train_RMSE_total": 3.0,
                "train_RMSE_X": 3.1,
                "train_RMSE_Y": 3.2,
                "train_RMSE_Z": 3.3,
            },
        },
        lr_stage_b=0.01,
        epochs_ran_stage_b=10,
    )
    assert row["schema_version"] == RUN_HISTORY_SCHEMA_VERSION
    assert row["RMSE_test_total"] == 3.0
    out = append_run_history(path, [row])
    assert len(out) == 1
    assert next_run_id(path) == 2
    row2 = dict(row)
    row2["run_id"] = 2
    row2["RMSE_test_total"] = 5.0
    append_run_history(path, [row2])
    assert next_run_id(path) == 3


def test_next_seed_continues_from_history(tmp_path: Path) -> None:
    path = tmp_path / "hist.csv"
    assert next_seed(path, default=0) == 0
    assert next_seed(path, default=7) == 7
    row = build_history_row(
        run_id=1,
        repeat_index=0,
        notebook_id="08_3A_2",
        experiment_id="exp",
        variant_id="v1",
        fusion_pattern="single_view",
        backbone_kind="fourier",
        metrics_by_split={
            "train": {
                "train_RMSE_total": 1.0,
                "train_RMSE_X": 1.0,
                "train_RMSE_Y": 1.0,
                "train_RMSE_Z": 1.0,
            },
            "validation": {
                "train_RMSE_total": 2.0,
                "train_RMSE_X": 2.0,
                "train_RMSE_Y": 2.0,
                "train_RMSE_Z": 2.0,
            },
            "test": {
                "train_RMSE_total": 3.0,
                "train_RMSE_X": 3.0,
                "train_RMSE_Y": 3.0,
                "train_RMSE_Z": 3.0,
            },
        },
        seed=0,
    )
    append_run_history(path, [row])
    row2 = dict(row)
    row2["run_id"] = 2
    row2["repeat_index"] = 1
    row2["seed"] = 2
    append_run_history(path, [row2])
    # Continues past max seed (2), not max run_id.
    assert next_seed(path, default=0) == 3


def test_aggregate_and_load_summary(tmp_path: Path) -> None:
    path = tmp_path / "hist.csv"
    rows = []
    for run_id, test_rmse in ((1, 1.0), (2, 3.0)):
        rows.append(
            build_history_row(
                run_id=run_id,
                repeat_index=run_id - 1,
                notebook_id="10_1B",
                experiment_id="exp",
                variant_id="m10_1c_e2e_illumination_fusion",
                fusion_pattern="e2e",
                backbone_kind="fourier",
                metrics_by_split={
                    "train": {
                        "train_RMSE_total": 0.5,
                        "train_RMSE_X": 0.5,
                        "train_RMSE_Y": 0.5,
                        "train_RMSE_Z": 0.5,
                    },
                    "validation": {
                        "train_RMSE_total": 0.8,
                        "train_RMSE_X": 0.8,
                        "train_RMSE_Y": 0.8,
                        "train_RMSE_Z": 0.8,
                    },
                    "test": {
                        "train_RMSE_total": test_rmse,
                        "train_RMSE_X": test_rmse,
                        "train_RMSE_Y": test_rmse,
                        "train_RMSE_Z": test_rmse,
                    },
                },
            )
        )
    append_run_history(path, rows)
    hist = pd.read_csv(path)
    summary = aggregate_run_history(hist)
    assert len(summary) == 1
    assert float(summary.iloc[0]["RMSE_test_mean"]) == 2.0
    assert float(summary.iloc[0]["RMSE_test_std"]) > 0.0
    assert float(summary.iloc[0]["RMSE_test_total_mean"]) == 2.0

    loaded, src = load_summary_for_plots(path, session_run_ids=[1, 2])
    assert loaded is not None
    assert "history" in src
    assert float(loaded.iloc[0]["n_runs"]) == 2
