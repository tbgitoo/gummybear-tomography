"""Unit tests for M9 frozen-fusion helpers (mocked training)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tomography_ml.studies.m9_frozen_fusion import (
    FOURIER_DISPLAY,
    DEFAULT_LR_STAGE_B_GRID,
    run_fusion_lr_study,
)
from tomography_ml_validation.plotting.m9_frozen_fusion import (
    combine_m9_comparisons,
    ensure_display_label,
    plot_m9_lr_study,
    plot_m9_param_counts_fourier_vs_pooled,
    plot_m9_rmse_fourier_vs_pooled,
    plot_m9_rmse_ladder,
)


class _ToyFusion:
    def __init__(self):
        import torch
        from torch import nn

        self.bias = nn.Parameter(torch.zeros(3))

    def parameters(self):
        return [self.bias]

    def to(self, _device):
        return self

    def eval(self):
        return self

    def state_dict(self):
        return {"bias": self.bias.detach().clone()}

    def load_state_dict(self, state):
        self.bias.data.copy_(state["bias"])

    def describe(self):
        return {"note": "toy", "learned_parameter_count": 3}

    def learned_parameter_count(self) -> int:
        return 3


def test_run_fusion_lr_study_fixed_and_sweep(monkeypatch) -> None:
    import tomography_ml.studies.m9_frozen_fusion as mod

    calls: list[float] = []

    def _fake_train(**kwargs):
        lr = float(kwargs["lr"])
        calls.append(lr)
        # Lower LR looks better so selection can be checked.
        return {
            "best_validation_RMSE_total": lr,
            "history": [{"train_loss": lr}],
            "parameter_count": 3,
        }

    monkeypatch.setattr(mod, "train_full_split", _fake_train)

    class _Loader:
        def __iter__(self):
            return iter([])

        def __len__(self):
            return 1

    fixed = run_fusion_lr_study(
        tag="toy",
        packing_label="mean_pool",
        make_model=_ToyFusion,
        train_loader=_Loader(),
        val_loader=_Loader(),
        batch_xy=lambda a, b: (a, b),
        device="cpu",
        lr_fixed=3e-3,
        lr_grid=DEFAULT_LR_STAGE_B_GRID,
        run_study=False,
        select_best_val_lr=True,
        num_epochs=2,
        early_stop_patience=1,
        verbose=False,
    )
    assert fixed["selection_mode"] == "fixed_lr"
    assert abs(fixed["selected_lr"] - 3e-3) < 1e-12
    assert len(calls) == 1

    calls.clear()
    swept = run_fusion_lr_study(
        tag="toy",
        packing_label="mean_pool",
        make_model=_ToyFusion,
        train_loader=_Loader(),
        val_loader=_Loader(),
        batch_xy=lambda a, b: (a, b),
        device="cpu",
        lr_fixed=3e-3,
        lr_grid=(1e-2, 1e-3),
        run_study=True,
        select_best_val_lr=True,
        num_epochs=2,
        early_stop_patience=1,
        verbose=False,
    )
    assert swept["selection_mode"] == "best_val_rmse"
    assert abs(swept["selected_lr"] - 1e-3) < 1e-12
    assert set(calls) >= {1e-2, 1e-3, 3e-3}


def test_m9_plot_helpers(tmp_path: Path) -> None:
    lr = pd.DataFrame(
        [
            {"lr": 1e-3, "packing": "mean_pool", "best_val_rmse": 1.2},
            {"lr": 3e-3, "packing": "mean_pool", "best_val_rmse": 0.9},
            {"lr": 1e-3, "packing": "ordered_concat", "best_val_rmse": 1.0},
            {"lr": 3e-3, "packing": "ordered_concat", "best_val_rmse": 0.8},
            {"lr": 1e-3, "packing": "deepsets_fourier", "best_val_rmse": 0.95},
            {"lr": 3e-3, "packing": "deepsets_fourier", "best_val_rmse": 0.7},
        ]
    )
    fig = plot_m9_lr_study(lr, family="fourier")
    assert fig is not None

    rows = []
    for split in ("validation", "test"):
        for i, (vid, label) in enumerate(
            [
                ("f1_single_view_reference", "SV ref"),
                ("f1_shared_xyz_mean_control", "xyz mean"),
                ("m09_1_compact_fusion_mlp_mean_pool_frozen_fourier", "mean-pool MLP"),
                ("m09_1_deepsets_fourier", "DeepSets Fourier"),
                ("m09_1_compact_fusion_mlp_frozen_fourier", "ordered concat"),
            ]
        ):
            rows.append(
                {
                    "variant_id": vid,
                    "display_label": label,
                    "split": split,
                    "RMSE_total": 1.0 - 0.1 * i,
                    "learned_parameter_count": 1000 * (i + 1),
                }
            )
    cmp_a = pd.DataFrame(rows)
    fig2 = plot_m9_rmse_ladder(cmp_a, family="fourier")
    assert fig2 is not None

    rows_b = []
    for split in ("validation", "test"):
        for i, label in enumerate(
            [
                "SV pooled",
                "xyz mean pooled",
                "mean-pool MLP pooled",
                "DeepSets no-Fourier",
                "ordered concat pooled",
            ]
        ):
            rows_b.append(
                {
                    "variant_id": f"b_{i}",
                    "display_label": label,
                    "split": split,
                    "RMSE_total": 2.0 - 0.1 * i,
                    "learned_parameter_count": 1000 * (i + 1),
                }
            )
    cmp_b = pd.DataFrame(rows_b)
    fig3 = plot_m9_param_counts_fourier_vs_pooled(cmp_a, cmp_b)
    assert fig3 is not None
    fig4 = plot_m9_rmse_fourier_vs_pooled(cmp_a, cmp_b, split="validation")
    assert fig4 is not None
    combined = combine_m9_comparisons(cmp_a, cmp_b)
    assert set(combined["family"]) == {"fourier", "pooled"}
    labeled = ensure_display_label(
        cmp_a.drop(columns=["display_label"]), FOURIER_DISPLAY
    )
    assert "SV ref" in set(labeled["display_label"])
