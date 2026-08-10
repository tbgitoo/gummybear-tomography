"""Unit tests for M9 e2e geometry-fusion helpers (mocked training)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from tomography_ml.studies.m9_e2e_geometry_fusion import (
    FOURIER_DISPLAY,
    POOLED_DISPLAY,
    M9E2EConfig,
)
from tomography_ml.studies.study_checkpoints import (
    M09_E2E_FOURIER_GEOMETRY_FUSION,
    M09_E2E_POOLED_GEOMETRY_FUSION,
)
from tomography_ml_validation.plotting.m9_e2e_geometry_fusion import (
    plot_m9_e2e_param_counts_fourier_vs_pooled,
    plot_m9_e2e_rmse_fourier_vs_pooled,
    plot_m9_e2e_rmse_ladder,
)


def _toy_comparison(family: str) -> pd.DataFrame:
    display = FOURIER_DISPLAY if family == "fourier" else POOLED_DISPLAY
    rows = []
    for split in ("validation", "test"):
        for i, (vid, label) in enumerate(display.items()):
            rows.append(
                {
                    "variant_id": vid,
                    "display_label": label,
                    "split": split,
                    "RMSE_total": 1.0 + 0.1 * i,
                    "learned_parameter_count": 1000 * (i + 1),
                }
            )
    return pd.DataFrame(rows)


def test_m9_e2e_plot_helpers() -> None:
    a = _toy_comparison("fourier")
    b = _toy_comparison("pooled")
    fig = plot_m9_e2e_rmse_ladder(a, family="fourier")
    assert fig is not None
    fig2 = plot_m9_e2e_param_counts_fourier_vs_pooled(a, b)
    assert fig2 is not None
    fig3 = plot_m9_e2e_rmse_fourier_vs_pooled(a, b, split="test")
    assert fig3 is not None


def test_run_m9_e2e_geometry_fusion_family_load_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    import tomography_ml.studies.m9_e2e_geometry_fusion as mod

    results = tmp_path / "results"
    results.mkdir()
    ckpt = results / M09_E2E_FOURIER_GEOMETRY_FUSION
    comparison = _toy_comparison("fourier")
    stage = {"w": torch.zeros(1)}
    torch.save(
        {
            "stage_a_state": stage,
            "f2_state": stage,
            "f3_state": stage,
            "comparison": comparison.to_dict(orient="records"),
        },
        ckpt,
    )

    class _Block:
        architecture = type("A", (), {"head_hidden": 8})()
        x_field = "img"
        image_normalize = "none"
        optical_setup_id_reference = "opt"

        def lr_by_role(self):
            return {"primary": 0.03, "negative_control": 0.001}

    class _ToyMod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.fusion_hidden = 128
            self.fusion_depth = 1

        def describe(self):
            return {"note": "toy"}

        def learned_parameter_count(self):
            return 1

        def fusion_parameter_count(self):
            return 1

    monkeypatch.setattr(mod, "m8_single_view_block_freeze", _Block)
    monkeypatch.setattr(mod, "load_catalog_jobs", lambda *a, **k: [])
    monkeypatch.setattr(mod, "build_catalog_rows", lambda jobs: [])
    monkeypatch.setattr(mod, "resolve_view_angles", lambda *a, **k: (0.0, 60.0))
    monkeypatch.setattr(
        mod, "new_frozen_single_view_expert", lambda **k: _ToyMod()
    )
    monkeypatch.setattr(
        mod.GeometryAwareFourierFusionLocalizer,
        "for_09_2",
        classmethod(lambda cls, *a, **k: _ToyMod()),
    )
    monkeypatch.setattr(
        mod.GeometryAwareFourierFusionLocalizer,
        "for_09_3",
        classmethod(lambda cls, *a, **k: _ToyMod()),
    )

    cfg = M9E2EConfig(
        family="fourier",
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
    out = mod.run_m9_e2e_geometry_fusion_family(cfg)
    assert out.skipped_train is True
    assert len(out.comparison_df) == len(comparison)
    assert out.checkpoint_path == ckpt
    assert M09_E2E_POOLED_GEOMETRY_FUSION.endswith(".pt")
