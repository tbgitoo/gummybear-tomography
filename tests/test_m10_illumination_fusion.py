"""Unit tests for M10 illumination-fusion study helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from tomography_ml.studies.m10_illumination_fusion import (
    M10IlluminationConfig,
    _artifact_names,
)
from tomography_ml.studies.study_checkpoints import (
    M10_E2E_ILLUMINATION_FUSION,
    M10_FROZEN_ILLUMINATION_FUSION,
)
from tomography_ml_validation.plotting.illumination_fusion import (
    PLOT_CONFIG_10_1A,
    plot_m10_backbone_rmse_ladder,
    plot_m10_param_counts_fourier_vs_pooled,
)


def test_m10_artifact_names_and_checkpoint_constants() -> None:
    a = _artifact_names("frozen")
    b = _artifact_names("e2e")
    assert a.checkpoint_name == M10_FROZEN_ILLUMINATION_FUSION
    assert b.checkpoint_name == M10_E2E_ILLUMINATION_FUSION
    assert a.csv_comparison.startswith("m10_1a_")
    assert b.csv_comparison.startswith("m10_1b_")


def test_m10_plot_helpers() -> None:
    rows = []
    for split in ("validation", "test"):
        for vid, kind, n in (
            ("m10_1a_single_illumination", "fourier", 100),
            ("m10_1b_mean_xyz_illuminations", "fourier", 100),
            ("m10_1a_c_frozen_illumination_fusion", "fourier", 200),
            ("m10_1a_d_frozen_illumination_angle_fusion", "fourier", 300),
            ("m10_1a_single_illumination_pooled", "pooled", 80),
            ("m10_1b_mean_xyz_illuminations_pooled", "pooled", 80),
            ("m10_1a_c_frozen_illumination_fusion_pooled", "pooled", 180),
            ("m10_1a_d_frozen_illumination_angle_fusion_pooled", "pooled", 280),
        ):
            rows.append(
                {
                    "variant_id": vid,
                    "backbone_kind": kind,
                    "split": split,
                    "RMSE_total": 1.0,
                    "learned_parameter_count": n,
                }
            )
    df = pd.DataFrame(rows)
    fig = plot_m10_backbone_rmse_ladder(
        df, config=PLOT_CONFIG_10_1A, backbone_kind="fourier"
    )
    assert fig is not None
    fig2 = plot_m10_backbone_rmse_ladder(
        df, config=PLOT_CONFIG_10_1A, backbone_kind="pooled"
    )
    assert fig2 is not None
    fig3 = plot_m10_param_counts_fourier_vs_pooled(df, config=PLOT_CONFIG_10_1A)
    assert fig3 is not None


def test_run_m10_illumination_fusion_load_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    import tomography_ml.studies.m10_illumination_fusion as mod

    results = tmp_path / "results"
    results.mkdir()
    names = _artifact_names("frozen")
    ckpt = results / names.checkpoint_name
    comparison = pd.DataFrame(
        [
            {
                "variant_id": "m10_1a_single_illumination",
                "backbone_kind": "fourier",
                "split": "test",
                "RMSE_total": 1.2,
            }
        ]
    )
    torch.save(
        {
            "comparison_last_run": comparison.to_dict(orient="records"),
            "session_summary": [],
            "lr_study": [],
            "selected_lrs": {"LR_STAGE_B_C_FOURIER": 0.003},
            "session_run_ids": [1],
        },
        ckpt,
    )

    monkeypatch.setattr(
        mod,
        "m8_single_view_block_freeze",
        lambda: type(
            "B",
            (),
            {
                "architecture": type("A", (), {"head_hidden": 8})(),
                "x_field": "img",
                "image_normalize": "none",
                "lr_by_role": lambda self: {
                    "primary": 0.03,
                    "negative_control": 0.001,
                },
            },
        )(),
    )

    cfg = M10IlluminationConfig(
        mode="frozen",
        workbook_path=tmp_path / "wb.xlsx",
        data_root=tmp_path,
        results_dir=results,
        stl_root=tmp_path,
        device="cpu",
        num_epochs=2,
        early_stop_patience=1,
        batch_size=4,
        load_existing=True,
        retrain=False,
        verbose=False,
    )
    out = mod.run_m10_illumination_fusion(cfg)
    assert out.skipped_train is True
    assert len(out.comparison_df) == 1
    assert out.checkpoint_path == ckpt
