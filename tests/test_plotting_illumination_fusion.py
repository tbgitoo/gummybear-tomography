"""Tests for shared illumination-fusion plotting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tomography_ml_validation.plotting import (
    PLOT_CONFIG_10_1A,
    PLOT_CONFIG_10_1B,
    plot_illumination_fusion_results,
)
from tomography_ml_validation.run_history import (
    append_run_history,
    build_history_row,
)


def _metrics(total: float) -> dict:
    return {
        "train_RMSE_total": total,
        "train_RMSE_X": total,
        "train_RMSE_Y": total,
        "train_RMSE_Z": total,
    }


def test_plot_configs_variant_ids() -> None:
    assert "m10_1a_single_illumination" in PLOT_CONFIG_10_1A.short_labels
    assert "m10_1a_single_illumination" in PLOT_CONFIG_10_1B.short_labels
    assert "m10_1c_e2e_illumination_fusion" in PLOT_CONFIG_10_1B.order_fourier
    assert "m10_1a_c_frozen_illumination_fusion" in PLOT_CONFIG_10_1A.order_fourier


def test_plot_illumination_fusion_results_smoke(tmp_path: Path) -> None:
    hist = tmp_path / "m10_1a_run_history.csv"
    rows = []
    for run_id, rmse in ((1, 1.0), (2, 1.4)):
        for vid in PLOT_CONFIG_10_1A.order_fourier:
            rows.append(
                build_history_row(
                    run_id=run_id,
                    repeat_index=run_id - 1,
                    notebook_id="10_1A",
                    experiment_id="exp",
                    variant_id=vid,
                    fusion_pattern="p",
                    backbone_kind="fourier",
                    metrics_by_split={
                        "train": _metrics(0.5),
                        "validation": _metrics(0.8),
                        "test": _metrics(rmse),
                    },
                )
            )
        # one pooled companion for pair plot
        rows.append(
            build_history_row(
                run_id=run_id,
                repeat_index=run_id - 1,
                notebook_id="10_1A",
                experiment_id="exp",
                variant_id="m10_1a_c_frozen_illumination_fusion_pooled",
                fusion_pattern="p",
                backbone_kind="pooled",
                metrics_by_split={
                    "train": _metrics(0.5),
                    "validation": _metrics(0.8),
                    "test": _metrics(rmse + 0.1),
                },
            )
        )
    append_run_history(hist, rows)
    out = plot_illumination_fusion_results(
        results_dir=tmp_path,
        config=PLOT_CONFIG_10_1A,
        history_path=hist,
        show=False,
    )
    assert out["summary_df"] is not None
    assert len(out["summary_df"]) >= 4
    assert (tmp_path / "m10_1a_rmse_total_barplot.png").is_file()
