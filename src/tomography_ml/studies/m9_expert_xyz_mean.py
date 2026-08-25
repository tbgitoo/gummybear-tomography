"""M9 09_0 — mean of per-view angle-specific expert coordinates.

Trains one WIN 3J Fourier expert per sampled orbit angle, then averages xyz
at evaluation time (no learned fusion). Diagnostic CSVs support bias-vs-spread
plots in the thin milestone notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gummybear.paths import display_path
from tomography_ml.gummybear_data_catalog import build_catalog_rows, load_catalog_jobs
from tomography_ml.gummybear_data_catalog.task_dataset import (
    DatasetTaskSpec,
    build_task_dataset,
)
from tomography_ml.localization.architecture_capability import (
    evaluate_split_rmse,
    per_axis_rmse,
    train_full_split,
)
from tomography_ml.localization.builders import m8_single_view_block_freeze
from tomography_ml.localization.localize_multiview import (
    FUSION_PATTERN_09_0,
    ExpertXyzMeanLocalizer,
    MeanLatentFusionLocalizer,
    new_frozen_single_view_expert,
    shared_xyz_mean,
)
from tomography_ml.studies.m9_frozen_fusion import resolve_view_angles
from tomography_ml.studies.study_checkpoints import (
    M09_EXPERT_XYZ_MEAN,
    clone_state_dict,
    load_study_checkpoint,
    save_study_checkpoint,
)
from tomography_ml.training.training_helpers import (
    make_batch_xy_multiview,
    make_batch_xy_single,
)

Y_FIELDS: tuple[str, ...] = ("particle_x", "particle_y", "particle_z")


@dataclass
class M9ExpertXyzMeanConfig:
    """Hyperparameters and paths for the 09_0 expert-mean study."""

    workbook_path: Path
    data_root: Path
    results_dir: Path
    stl_root: Path
    device: torch.device | str
    num_epochs: int = 200
    early_stop_patience: int = 40
    batch_size: int = 16
    angle_stride_deg: float = 10.0
    load_existing: bool = True
    retrain: bool = False
    verbose: bool = True


@dataclass
class M9ExpertXyzMeanResult:
    """Artifacts from :func:`run_m9_expert_xyz_mean`."""

    comparison_df: pd.DataFrame
    experts_df: pd.DataFrame
    view_angles: tuple[float, ...]
    sv_angle: float
    results_dir: Path
    comparison_path: Path
    experts_path: Path
    checkpoint_path: Path
    skipped_train: bool
    bank: ExpertXyzMeanLocalizer | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def assert_affine_identity_shared_linear(*, hidden: int = 128, n_views: int = 6) -> float:
    """Return max abs error of mean(Linear(h)) vs Linear(mean(h)) on noise."""
    torch.manual_seed(0)
    backbone = new_frozen_single_view_expert(n_outputs=3, hidden=int(hidden))
    mean_latent = MeanLatentFusionLocalizer(backbone, freeze_encoder=False)
    views = torch.randn(4, n_views, 1, 32, 32)
    with torch.no_grad():
        delta = (shared_xyz_mean(backbone, views) - mean_latent(views)).abs().max()
    return float(delta)


def _make_split_dataset(
    catalog_rows,
    *,
    split: str,
    keep_angles_deg: float | Sequence[float],
    x_field: str,
    image_normalize: str,
    optical_setup_id: str,
) -> Dataset:
    task = DatasetTaskSpec(
        name=f"m9_0_{split}_{keep_angles_deg}",
        row_filter={
            "field_status": "complete",
            "optical_setup_id": optical_setup_id,
            "split": split,
        },
        x_fields=(x_field,),
        y_fields=Y_FIELDS,
        keep_angles_deg=keep_angles_deg,
        image_normalize=image_normalize,
    )
    return build_task_dataset(catalog_rows, task)


def run_m9_expert_xyz_mean(cfg: M9ExpertXyzMeanConfig) -> M9ExpertXyzMeanResult:
    """Train (or load) per-angle experts and score xyz-mean vs a single-view ref."""
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device
    comparison_path = results_dir / "m09_0_comparison.csv"
    experts_path = results_dir / "m09_0_per_angle_experts.csv"
    checkpoint_path = results_dir / M09_EXPERT_XYZ_MEAN

    block = m8_single_view_block_freeze()
    arch = block.architecture
    x_field = str(block.x_field)
    image_normalize = str(block.image_normalize)
    optical_id = str(block.optical_setup_id_reference)
    lr = float(block.lr_by_role()["primary"])

    catalog_jobs = load_catalog_jobs(cfg.workbook_path, cfg.data_root, stl_root=cfg.stl_root)
    catalog_rows = build_catalog_rows(catalog_jobs)
    view_angles = resolve_view_angles(
        catalog_rows,
        optical_setup_id=optical_id,
        angle_stride_deg=cfg.angle_stride_deg,
    )
    sv_angle = 180.0 if 180.0 in view_angles else float(view_angles[len(view_angles) // 2])

    skip_train = bool(cfg.load_existing and checkpoint_path.is_file() and not cfg.retrain)
    if cfg.verbose:
        print(
            f"M9 09_0: views={view_angles}  epochs={cfg.num_epochs}  "
            f"patience={cfg.early_stop_patience}  SKIP_TRAIN={skip_train}",
            flush=True,
        )

    batch_xy_single = make_batch_xy_single(x_field=x_field, y_fields=Y_FIELDS, device=device)
    batch_xy_multiview = make_batch_xy_multiview(
        x_field=x_field, y_fields=Y_FIELDS, device=device
    )

    def make_split(split: str, keep: float | Sequence[float]) -> Dataset:
        return _make_split_dataset(
            catalog_rows,
            split=split,
            keep_angles_deg=keep,
            x_field=x_field,
            image_normalize=image_normalize,
            optical_setup_id=optical_id,
        )

    expert_models: dict[float, nn.Module] = {}
    expert_rows: list[dict[str, Any]] = []

    if skip_train:
        payload = load_study_checkpoint(checkpoint_path)
        stored_angles = tuple(float(a) for a in payload["expert_angles_deg"])
        states: dict[str, Any] = payload["expert_state_dicts"]
        for theta in stored_angles:
            model = new_frozen_single_view_expert(
                n_outputs=3, hidden=int(arch.head_hidden)
            ).to(device)
            key = f"{float(theta):.4f}"
            model.load_state_dict(states[key])
            expert_models[float(theta)] = model
        view_angles = stored_angles
        sv_angle = float(payload.get("sv_angle", sv_angle))
        experts_df = pd.read_csv(experts_path) if experts_path.is_file() else pd.DataFrame()
        comparison_df = (
            pd.read_csv(comparison_path) if comparison_path.is_file() else pd.DataFrame()
        )
        bank = ExpertXyzMeanLocalizer(expert_models).to(device)
        extra = {
            "x_field": x_field,
            "catalog_rows": catalog_rows,
            "block": block,
        }
        if cfg.verbose:
            print(f"Loaded {display_path(checkpoint_path)}")
        return M9ExpertXyzMeanResult(
            comparison_df=comparison_df,
            experts_df=experts_df,
            view_angles=view_angles,
            sv_angle=sv_angle,
            results_dir=results_dir,
            comparison_path=comparison_path,
            experts_path=experts_path,
            checkpoint_path=checkpoint_path,
            skipped_train=True,
            bank=bank,
            extra=extra,
        )

    for theta in view_angles:
        train_ds = make_split("train", theta)
        val_ds = make_split("validation", theta)
        train_loader = DataLoader(
            train_ds, batch_size=min(int(cfg.batch_size), len(train_ds)), shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=min(int(cfg.batch_size), len(val_ds)), shuffle=False
        )
        model = new_frozen_single_view_expert(
            n_outputs=3, hidden=int(arch.head_hidden)
        ).to(device)
        if cfg.verbose:
            print(f"=== expert θ={theta:g}°  train={len(train_ds)} val={len(val_ds)} ===")
        fit = train_full_split(
            model=model,
            train_loader=train_loader,
            batch_xy=batch_xy_single,
            device=device,
            num_epochs=int(cfg.num_epochs),
            lr=lr,
            val_loader=val_loader,
            y_fields=Y_FIELDS,
            early_stop_patience=int(cfg.early_stop_patience),
        )
        train_m = evaluate_split_rmse(
            model=model,
            loader=train_loader,
            batch_xy=batch_xy_single,
            y_fields=Y_FIELDS,
            prefix="train",
        )
        val_m = evaluate_split_rmse(
            model=model,
            loader=val_loader,
            batch_xy=batch_xy_single,
            y_fields=Y_FIELDS,
            prefix="validation",
        )
        expert_rows.append(
            {
                "angle_deg": float(theta),
                "n_train": len(train_ds),
                "n_validation": len(val_ds),
                "learned_parameter_count": int(fit["parameter_count"]),
                "epochs_ran": len(fit["history"]),
                **train_m,
                **val_m,
            }
        )
        expert_models[float(theta)] = model

    experts_df = pd.DataFrame(expert_rows).sort_values("angle_deg").reset_index(drop=True)
    experts_df.to_csv(experts_path, index=False)

    bank = ExpertXyzMeanLocalizer(expert_models).to(device)
    sv_model = expert_models[sv_angle]

    def eval_sv(split: str) -> dict[str, float]:
        ds = make_split(split, sv_angle)
        loader = DataLoader(ds, batch_size=min(int(cfg.batch_size), len(ds)), shuffle=False)
        return evaluate_split_rmse(
            model=sv_model,
            loader=loader,
            batch_xy=batch_xy_single,
            y_fields=Y_FIELDS,
            prefix="train",
        )

    def eval_bank(split: str) -> dict[str, float]:
        ds = make_split(split, list(view_angles))
        loader = DataLoader(ds, batch_size=min(int(cfg.batch_size), len(ds)), shuffle=False)
        preds, tgts = [], []
        bank.eval()
        with torch.no_grad():
            for images, targets in loader:
                views, batch_targets = batch_xy_multiview(images, targets)
                pred = bank(views, list(view_angles))
                preds.append(pred.detach().cpu())
                tgts.append(batch_targets.detach().cpu())
        return per_axis_rmse(torch.cat(preds), torch.cat(tgts), Y_FIELDS)

    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        sv = eval_sv(split)
        fused = eval_bank(split)
        rows.append(
            {
                "variant_id": "f0_single_view_reference",
                "fusion_pattern": "single_view_reference",
                "n_views": 1,
                "keep_angles_deg": sv_angle,
                "split": split,
                "RMSE_total": sv["train_RMSE_total"],
                "RMSE_X": sv["train_RMSE_X"],
                "RMSE_Y": sv["train_RMSE_Y"],
                "RMSE_Z": sv["train_RMSE_Z"],
                "learned_parameter_count": int(
                    sum(p.numel() for p in sv_model.parameters() if p.requires_grad)
                ),
            }
        )
        rows.append(
            {
                "variant_id": "m09_0_expert_xyz_mean",
                "fusion_pattern": FUSION_PATTERN_09_0,
                "n_views": len(view_angles),
                "keep_angles_deg": None,
                "split": split,
                "RMSE_total": fused["train_RMSE_total"],
                "RMSE_X": fused["train_RMSE_X"],
                "RMSE_Y": fused["train_RMSE_Y"],
                "RMSE_Z": fused["train_RMSE_Z"],
                "learned_parameter_count": bank.learned_parameter_count(),
            }
        )
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(comparison_path, index=False)

    save_study_checkpoint(
        checkpoint_path,
        {
            "expert_angles_deg": list(view_angles),
            "sv_angle": float(sv_angle),
            "expert_state_dicts": {
                f"{float(theta):.4f}": clone_state_dict(model)
                for theta, model in expert_models.items()
            },
        },
    )
    if cfg.verbose:
        print(f"Wrote {display_path(comparison_path)}")
        print(comparison_df.to_string(index=False))

    return M9ExpertXyzMeanResult(
        comparison_df=comparison_df,
        experts_df=experts_df,
        view_angles=view_angles,
        sv_angle=sv_angle,
        results_dir=results_dir,
        comparison_path=comparison_path,
        experts_path=experts_path,
        checkpoint_path=checkpoint_path,
        skipped_train=False,
        bank=bank,
        extra={"x_field": x_field, "catalog_rows": catalog_rows, "block": block},
    )
