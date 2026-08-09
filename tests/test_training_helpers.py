"""Unit tests for tomography_ml.training.training_helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn

from tomography_ml.training.training_helpers import (
    batch_from_indices,
    collect_prediction_errors,
    eval_stack,
    lr_close,
    make_batch_xy_single,
    persist_stage_b_lr_artifacts,
    resolve_run_lr_study,
    run_stage_b_lr_study,
    train_e2e,
)


def test_lr_close() -> None:
    assert lr_close(1e-3, 1e-3)
    assert not lr_close(1e-3, 2e-3)


class _ToyStack(torch.utils.data.Dataset):
    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        views = torch.zeros(2, 1, 4, 4) + float(index)
        targets = {"particle_x": 1.0, "particle_y": 2.0, "particle_z": 3.0}
        lights = torch.tensor([0.0, 60.0])
        return views, targets, lights


class _ToyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, views, angles_deg=None):
        # Constant prediction shaped [B, 3]
        b = views.shape[0]
        base = torch.tensor([1.0, 2.0, 3.0], device=views.device)
        return base.unsqueeze(0).expand(b, -1) + self.bias

    def describe(self):
        return {"note": "toy", "learned_parameter_count": 3}

    def learned_parameter_count(self) -> int:
        return 3


def test_batch_from_indices_and_eval_stack() -> None:
    ds = _ToyStack(4)
    device = torch.device("cpu")
    y = ("particle_x", "particle_y", "particle_z")
    views, targets, lights = batch_from_indices(ds, [0, 2], device, y)
    assert views.shape == (2, 2, 1, 4, 4)
    assert targets.shape == (2, 3)
    assert lights.shape == (2, 2)

    metrics = eval_stack(
        ds,
        lambda v, _l: torch.tensor([[1.0, 2.0, 3.0]]).expand(v.shape[0], -1),
        y_fields=y,
        device=device,
        batch_size=2,
    )
    assert metrics["train_RMSE_total"] == 0.0


def test_make_batch_xy_single() -> None:
    batch_xy = make_batch_xy_single(
        x_field="anomaly_ref",
        y_fields=("particle_x", "particle_y", "particle_z"),
        device="cpu",
    )
    images = {"anomaly_ref": torch.zeros(2, 1, 1, 4, 4)}
    targets = {
        "particle_x": torch.tensor([1.0, 1.0]),
        "particle_y": torch.tensor([2.0, 2.0]),
        "particle_z": torch.tensor([3.0, 3.0]),
    }
    views, y = batch_xy(images, targets)
    assert views.shape == (2, 1, 4, 4)
    assert y.shape == (2, 3)


def test_make_batch_xy_multiview() -> None:
    from tomography_ml.training.training_helpers import make_batch_xy_multiview

    batch_xy = make_batch_xy_multiview(
        x_field="anomaly_ref",
        y_fields=("particle_x", "particle_y", "particle_z"),
        device="cpu",
    )
    images = {"anomaly_ref": torch.zeros(2, 6, 1, 4, 4)}
    targets = {
        "particle_x": torch.tensor([1.0, 1.0]),
        "particle_y": torch.tensor([2.0, 2.0]),
        "particle_z": torch.tensor([3.0, 3.0]),
    }
    views, y = batch_xy(images, targets)
    assert views.shape == (2, 6, 1, 4, 4)
    assert y.shape == (2, 3)
    # Missing V axis → unsqueeze.
    images4 = {"anomaly_ref": torch.zeros(2, 1, 4, 4)}
    views4, _ = batch_xy(images4, targets)
    assert views4.shape == (2, 1, 1, 4, 4)


def test_collect_prediction_errors() -> None:
    class _ToySV(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int):
            images = {"x": torch.zeros(1, 1, 4, 4)}
            targets = {
                "particle_x": torch.tensor(1.0),
                "particle_y": torch.tensor(2.0),
                "particle_z": torch.tensor(3.0),
            }
            return images, targets

    class _Const(nn.Module):
        def forward(self, views):
            b = views.shape[0]
            return torch.tensor([[1.0, 2.0, 3.0]]).expand(b, -1)

    batch_xy = make_batch_xy_single(
        x_field="x",
        y_fields=("particle_x", "particle_y", "particle_z"),
        device="cpu",
    )
    loader = torch.utils.data.DataLoader(_ToySV(), batch_size=2)
    errs, mse = collect_prediction_errors(_Const(), loader, batch_xy)
    assert errs.shape == (4,)
    assert float(errs.max()) == 0.0
    assert mse == 0.0


def test_train_e2e_and_lr_study() -> None:
    ds = _ToyStack(6)
    y = ("particle_x", "particle_y", "particle_z")
    device = torch.device("cpu")
    model = _ToyFusion()
    fit = train_e2e(
        model,
        ds,
        ds,
        y_fields=y,
        device=device,
        use_angles=False,
        lr=1e-2,
        num_epochs=2,
        early_stop_patience=5,
        batch_size=4,
        progress_label="toy",
    )
    assert fit["best_val_rmse"] == 0.0
    assert len(fit["history"]) == 2

    out = run_stage_b_lr_study(
        tag="toy",
        make_model=_ToyFusion,
        train_ds=ds,
        val_ds=ds,
        y_fields=y,
        device=device,
        use_angles=False,
        lr_fixed=1e-2,
        run_study=True,
        select_best_val_lr=True,
        lr_grid=(1e-2, 1e-3),
        num_epochs=1,
        early_stop_patience=2,
        batch_size=4,
    )
    assert out["selected_lr"] in (1e-2, 1e-3)
    assert len(out["study_df"]) == 2


def test_persist_stage_b_lr_artifacts(tmp_path: Path) -> None:
    fourier = pd.DataFrame(
        [{"lr": 1e-3, "variant_tag": "C_Fourier", "best_val_rmse": 0.5}]
    )
    pooled = pd.DataFrame(
        [{"lr": 1e-3, "variant_tag": "C_pooled", "best_val_rmse": 0.6}]
    )
    recommended = {
        "LR_STAGE_B_C_FOURIER": 1e-3,
        "LR_STAGE_B_D_FOURIER": 2e-3,
        "LR_STAGE_B_C_POOLED": 3e-3,
        "LR_STAGE_B_D_POOLED": 4e-3,
    }
    combined = persist_stage_b_lr_artifacts(
        tmp_path,
        lr_study_fourier_df=fourier,
        lr_study_pooled_df=pooled,
        include_pooled_control=True,
        recommended=recommended,
        csv_lr_study="lr_study.csv",
        csv_lr_study_fourier="lr_fourier.csv",
        csv_lr_study_pooled="lr_pooled.csv",
        json_recommended_lrs="recommended.json",
    )
    assert len(combined) == 2
    assert (tmp_path / "lr_study.csv").is_file()
    assert (tmp_path / "recommended.json").is_file()


def test_resolve_run_lr_study_if_unknown(tmp_path: Path) -> None:
    from tomography_ml.training.training_helpers import resolve_run_lr_study

    defaults = {
        "LR_STAGE_B_C_FOURIER": 0.01,
        "LR_STAGE_B_D_FOURIER": 0.01,
        "LR_STAGE_B_C_POOLED": 0.01,
        "LR_STAGE_B_D_POOLED": 0.01,
    }
    path = tmp_path / "recommended.json"

    # No file → all unknown under if_unknown.
    out = resolve_run_lr_study(
        "if_unknown",
        recommended_path=path,
        use_angle_film=True,
        include_pooled_control=True,
        lr_defaults=defaults,
    )
    assert out["effective_run_lr_study"] is True
    assert set(out["unknown_keys"]) == set(defaults)
    assert all(out["study_by_key"].values())

    # Legacy concat JSON: C keys reusable with film=True; D keys not.
    path.write_text(
        '{"LR_STAGE_B_C_FOURIER": 0.003, "LR_STAGE_B_D_FOURIER": 0.003, '
        '"LR_STAGE_B_C_POOLED": 0.003, "LR_STAGE_B_D_POOLED": 0.01}'
    )
    out_film = resolve_run_lr_study(
        "if_unknown",
        recommended_path=path,
        use_angle_film=True,
        include_pooled_control=True,
        lr_defaults=defaults,
    )
    assert out_film["lrs"]["LR_STAGE_B_C_FOURIER"] == 0.003
    assert out_film["study_by_key"]["LR_STAGE_B_C_FOURIER"] is False
    assert out_film["study_by_key"]["LR_STAGE_B_D_FOURIER"] is True
    assert out_film["effective_run_lr_study"] is True

    # Matching film fingerprint → skip all.
    path.write_text(
        '{"LR_STAGE_B_C_FOURIER": 0.003, "LR_STAGE_B_D_FOURIER": 0.005, '
        '"LR_STAGE_B_C_POOLED": 0.003, "LR_STAGE_B_D_POOLED": 0.01, '
        '"use_angle_film": true}'
    )
    out_known = resolve_run_lr_study(
        "if_unknown",
        recommended_path=path,
        use_angle_film=True,
        include_pooled_control=True,
        lr_defaults=defaults,
    )
    assert out_known["effective_run_lr_study"] is False
    assert out_known["lrs"]["LR_STAGE_B_D_FOURIER"] == 0.005
    assert not any(out_known["study_by_key"].values())

    # Force always / never.
    assert resolve_run_lr_study(
        True,
        recommended_path=path,
        use_angle_film=True,
        include_pooled_control=True,
        lr_defaults=defaults,
    )["effective_run_lr_study"] is True
    assert resolve_run_lr_study(
        False,
        recommended_path=path,
        use_angle_film=True,
        include_pooled_control=True,
        lr_defaults=defaults,
    )["effective_run_lr_study"] is False
