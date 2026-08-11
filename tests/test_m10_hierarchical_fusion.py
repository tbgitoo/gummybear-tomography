"""Unit tests for M10 hierarchical fusion helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from tomography_ml.localization.localize_multiview import (
    FUSION_PATTERN_10_2_POOLED,
    HierarchicalLightThenCameraFusionLocalizer,
    new_frozen_pooled_single_view_expert,
)
from tomography_ml.studies.m10_hierarchical_fusion import (
    M10HierarchicalConfig,
    CSV_COMPARISON,
)
from tomography_ml.studies.study_checkpoints import M10_HIERARCHICAL_LIGHT_THEN_CAMERA
from tomography_ml_validation.plotting.illumination_fusion import (
    plot_m10_hierarchical_lr_study,
    plot_m10_hierarchical_rmse_fourier_vs_pooled,
)


def test_for_10_2_pooled_forward() -> None:
    cams = (0.0, 90.0)
    lights = (0.0, 120.0, 240.0)
    trunk = new_frozen_pooled_single_view_expert(n_outputs=3, embed_dim=32)
    model = HierarchicalLightThenCameraFusionLocalizer.for_10_2_pooled(
        trunk,
        n_cameras=len(cams),
        n_lights=len(lights),
        camera_angles_deg=cams,
        light_angles_deg=lights,
        fusion_hidden=16,
        fusion_depth=1,
        camera_latent_dim=16,
    )
    assert model.backbone_kind == "pooled"
    assert model.fusion_pattern == FUSION_PATTERN_10_2_POOLED
    b = 2
    views = torch.randn(b, len(lights), len(cams), 1, 8, 8)
    out = model(views)
    assert out.shape == (b, 3)
    assert "pooled" in model.describe()["variant_id"]


def test_m10_hierarchical_plot_helpers() -> None:
    lr = pd.DataFrame(
        [
            {"lr": 1e-3, "best_val_rmse": 1.0, "backbone_kind": "fourier"},
            {"lr": 3e-4, "best_val_rmse": 0.8, "backbone_kind": "fourier"},
            {"lr": 1e-3, "best_val_rmse": 1.2, "backbone_kind": "pooled"},
            {"lr": 3e-4, "best_val_rmse": 1.1, "backbone_kind": "pooled"},
        ]
    )
    assert plot_m10_hierarchical_lr_study(lr) is not None
    rows = []
    for split in ("validation", "test"):
        for vid, rmse in (
            ("m10_2_single_view_reference", 1.5),
            ("m10_2_shared_xyz_mean_joint", 1.2),
            ("m10_2_hierarchical_light_then_camera", 0.9),
            ("m10_2_single_view_pooled_reference", 1.8),
            ("m10_2_shared_xyz_mean_pooled", 1.5),
            ("m10_2_hierarchical_pooled_light_then_camera", 1.3),
        ):
            rows.append(
                {
                    "variant_id": vid,
                    "split": split,
                    "RMSE_total": rmse,
                }
            )
    assert plot_m10_hierarchical_rmse_fourier_vs_pooled(pd.DataFrame(rows)) is not None


def test_run_m10_hierarchical_load_checkpoint(tmp_path: Path, monkeypatch) -> None:
    import tomography_ml.studies.m10_hierarchical_fusion as mod

    results = tmp_path / "results"
    results.mkdir()
    ckpt = results / M10_HIERARCHICAL_LIGHT_THEN_CAMERA
    comparison = pd.DataFrame(
        [
            {
                "variant_id": "m10_2_hierarchical_light_then_camera",
                "backbone_kind": "fourier",
                "split": "test",
                "RMSE_total": 0.9,
            }
        ]
    )
    torch.save(
        {
            "comparison": comparison.to_dict(orient="records"),
            "lr_study": [{"lr": 3e-4, "best_val_rmse": 0.9, "backbone_kind": "fourier"}],
            "selected_lrs": {"fourier": 3e-4, "pooled": 3e-4},
            "view_angles_deg": [0.0, 90.0],
            "light_angles_deg": [0.0, 120.0, 240.0],
        },
        ckpt,
    )

    class _Block:
        architecture = type("A", (), {"head_hidden": 8})()
        x_field = "img"
        image_normalize = "none"
        batch_size = 4

        def lr_by_role(self):
            return {"primary": 0.03, "negative_control": 0.001}

    monkeypatch.setattr(mod, "m8_single_view_block_freeze", _Block)

    cfg = M10HierarchicalConfig(
        workbook_path=tmp_path / "wb.xlsx",
        data_root=tmp_path,
        results_dir=results,
        stl_root=tmp_path,
        device="cpu",
        num_epochs=2,
        early_stop_patience=1,
        load_existing=True,
        retrain=False,
        verbose=False,
    )
    out = mod.run_m10_hierarchical_fusion(cfg)
    assert out.skipped_train is True
    assert len(out.comparison_df) == 1
    assert out.checkpoint_path.name == M10_HIERARCHICAL_LIGHT_THEN_CAMERA
    assert CSV_COMPARISON.endswith(".csv")
