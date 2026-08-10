"""M9 frozen multi-view fusion ladder (09_1A Fourier + 09_1B pooled).

Stage A trains a single-view trunk; Stage B freezes it and trains fusion heads
(mean-pool MLP, DeepSets, ordered concat) with an optional Stage-B LR sweep.
Matches ``notebooks/09_1A_frozen_fourier_fusion.ipynb`` /
``notebooks/09_1B_frozen_pooled_fusion.ipynb``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from gummybear.paths import display_path
from tomography_ml.gummybear_data_catalog import build_catalog_rows, load_catalog_jobs
from tomography_ml.gummybear_data_catalog.catalog import CatalogRow
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
    FUSION_PATTERN_09_1,
    FUSION_PATTERN_09_1_DEEPSETS_FOURIER,
    FUSION_PATTERN_09_1_DEEPSETS_NO_FOURIER,
    FUSION_PATTERN_09_1_MEAN_POOL,
    FUSION_PATTERN_09_1_MEAN_POOL_POOLED,
    FUSION_PATTERN_09_1_POOLED,
    M9_1_DEEPSETS_PHI_HIDDEN,
    M9_1_DEEPSETS_RHO_HIDDEN,
    PACKING_MEAN_POOL,
    PACKING_ORDERED_CONCAT,
    CompactLatentFusionLocalizer,
    FrozenEncoderDeepSetsLocalizer,
    new_frozen_pooled_single_view_expert,
    new_frozen_single_view_expert,
    shared_xyz_mean,
)
from tomography_ml.training.training_helpers import (
    lr_close,
    make_batch_xy_multiview,
    make_batch_xy_single,
)

Family = Literal["fourier", "pooled"]
MakeModelFn = Callable[[], nn.Module]

Y_FIELDS: tuple[str, ...] = ("particle_x", "particle_y", "particle_z")
DEFAULT_LR_STAGE_B_GRID: tuple[float, ...] = (
    0.03,
    0.01,
    0.003,
    1e-3,
    3e-4,
    1e-4,
    3e-5,
    1e-5,
)
DEFAULT_LR_STAGE_B: float = 3e-3
FUSION_HIDDEN: int = 128

FOURIER_VARIANT_ORDER: tuple[str, ...] = (
    "f1_single_view_reference",
    "f1_shared_xyz_mean_control",
    "m09_1_compact_fusion_mlp_mean_pool_frozen_fourier",
    "m09_1_deepsets_fourier",
    "m09_1_compact_fusion_mlp_frozen_fourier",
)
POOLED_VARIANT_ORDER: tuple[str, ...] = (
    "f1b_single_view_pooled_reference",
    "f1b_shared_xyz_mean_pooled_control",
    "m09_1_compact_fusion_mlp_mean_pool_frozen_pooled",
    "m09_1_deepsets_no_fourier",
    "m09_1_compact_fusion_mlp_frozen_pooled",
)

FOURIER_DISPLAY: dict[str, str] = {
    "f1_single_view_reference": "SV ref",
    "f1_shared_xyz_mean_control": "xyz mean",
    "m09_1_compact_fusion_mlp_mean_pool_frozen_fourier": "mean-pool MLP",
    "m09_1_deepsets_fourier": "DeepSets Fourier",
    "m09_1_compact_fusion_mlp_frozen_fourier": "ordered concat",
}
POOLED_DISPLAY: dict[str, str] = {
    "f1b_single_view_pooled_reference": "SV pooled",
    "f1b_shared_xyz_mean_pooled_control": "xyz mean pooled",
    "m09_1_compact_fusion_mlp_mean_pool_frozen_pooled": "mean-pool MLP pooled",
    "m09_1_deepsets_no_fourier": "DeepSets no-Fourier",
    "m09_1_compact_fusion_mlp_frozen_pooled": "ordered concat pooled",
}


@dataclass
class M9FusionConfig:
    """Hyperparameters and paths for one M9 frozen-fusion family run."""

    family: Family
    workbook_path: Path
    data_root: Path
    results_dir: Path
    stl_root: Path
    device: torch.device | str
    num_epochs: int = 200
    early_stop_patience: int = 40
    batch_size: int = 16
    angle_stride_deg: float = 60.0
    run_lr_study: bool = True
    lr_stage_b_grid: Sequence[float] = DEFAULT_LR_STAGE_B_GRID
    lr_stage_b_mean_pool: float = DEFAULT_LR_STAGE_B
    lr_stage_b_ordered_concat: float = DEFAULT_LR_STAGE_B
    lr_stage_b_deepsets: float = DEFAULT_LR_STAGE_B
    compact_select_best_val_lr: bool = False
    deepsets_select_best_val_lr: bool = True
    load_existing: bool = True
    retrain: bool = False
    fusion_hidden: int = FUSION_HIDDEN
    phi_hidden: int = M9_1_DEEPSETS_PHI_HIDDEN
    rho_hidden: int = M9_1_DEEPSETS_RHO_HIDDEN
    verbose: bool = True


@dataclass
class M9FusionFamilyResult:
    """Artifacts from :func:`run_m9_frozen_fusion_family`."""

    family: Family
    comparison_df: pd.DataFrame
    lr_study_df: pd.DataFrame
    view_angles: tuple[float, ...]
    lr_stage_a: float
    selected_lrs: dict[str, float]
    results_dir: Path
    comparison_path: Path
    lr_study_path: Path
    checkpoint_path: Path
    skipped_train: bool
    num_epochs: int
    early_stop_patience: int
    batch_size: int
    extra: dict[str, Any] = field(default_factory=dict)


def resolve_view_angles(
    catalog_rows: Sequence[CatalogRow],
    *,
    optical_setup_id: str,
    angle_stride_deg: float,
) -> tuple[float, ...]:
    """Subsample catalog orbit angles onto a regular stride (default 60° → V=6)."""
    orbit = tuple(
        float(a)
        for a in next(
            r.angles_deg
            for r in catalog_rows
            if r.field_status == "complete" and r.optical_setup_id == optical_setup_id
        )
    )
    views = tuple(
        sorted(
            {
                a
                for a in orbit
                if abs((a / float(angle_stride_deg)) - round(a / float(angle_stride_deg)))
                < 1e-6
            }
        )
    )
    if len(views) < 2:
        raise ValueError(f"Need ≥2 views; got {views} from orbit={orbit}")
    return views


def _make_split_dataset(
    catalog_rows: Sequence[CatalogRow],
    *,
    split: str,
    keep_angles_deg: float | Sequence[float],
    x_field: str,
    image_normalize: str,
    optical_setup_id: str,
    family: Family,
) -> Dataset:
    task = DatasetTaskSpec(
        name=f"m9_{family}_{split}_{keep_angles_deg}",
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


def run_fusion_lr_study(
    *,
    tag: str,
    packing_label: str,
    make_model: MakeModelFn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_xy: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str,
    lr_fixed: float,
    lr_grid: Sequence[float],
    run_study: bool,
    select_best_val_lr: bool,
    num_epochs: int,
    early_stop_patience: int,
    progress_prefix: str = "Stage B",
    verbose: bool = True,
) -> dict[str, Any]:
    """Stage-B LR sweep or single fixed-LR train (``train_full_split`` path)."""
    if run_study:
        lr_candidates = tuple(float(x) for x in lr_grid)
        if not any(lr_close(x, lr_fixed) for x in lr_candidates):
            lr_candidates = lr_candidates + (float(lr_fixed),)
    else:
        lr_candidates = (float(lr_fixed),)

    lr_rows: list[dict[str, Any]] = []
    states_by_lr: dict[float, dict[str, torch.Tensor]] = {}
    fits_by_lr: dict[float, dict[str, Any]] = {}
    describes_by_lr: dict[float, Any] = {}
    study_best_lr: float | None = None
    study_best_val = float("inf")

    for i, lr_b in enumerate(lr_candidates):
        if verbose:
            print(
                f"\n=== {progress_prefix} {tag}  lr={lr_b:g}  "
                f"({i + 1}/{len(lr_candidates)}) ===",
                flush=True,
            )
        model_b = make_model()
        if verbose and i == 0:
            print(model_b.describe())
        fit_b = train_full_split(
            model=model_b,
            train_loader=train_loader,
            batch_xy=batch_xy,
            device=device,
            num_epochs=int(num_epochs),
            lr=float(lr_b),
            val_loader=val_loader,
            y_fields=Y_FIELDS,
            early_stop_patience=int(early_stop_patience),
            progress_label=f"{progress_prefix} {tag} lr={lr_b:g}",
        )
        best_val = float(fit_b["best_validation_RMSE_total"])
        final_train = (
            float(fit_b["history"][-1]["train_loss"])
            if fit_b["history"]
            else float("nan")
        )
        lr_rows.append(
            {
                "lr": lr_b,
                "packing": packing_label,
                "variant_tag": tag,
                "best_val_rmse": best_val,
                "epochs_ran": len(fit_b["history"]),
                "final_train_loss": final_train,
                "converged_hint": bool(best_val < 2.0 and final_train < 10.0),
                "trainable": int(model_b.learned_parameter_count()),
            }
        )
        states_by_lr[lr_b] = {
            k: v.detach().cpu().clone() for k, v in model_b.state_dict().items()
        }
        fits_by_lr[lr_b] = fit_b
        describes_by_lr[lr_b] = model_b.describe()
        if verbose:
            print(
                f"  → lr={lr_b:g} best_val={best_val:.4f} "
                f"epochs={len(fit_b['history'])} final_train_loss={final_train:.4f}",
                flush=True,
            )
        if best_val < study_best_val:
            study_best_val = best_val
            study_best_lr = lr_b

    if select_best_val_lr and run_study:
        selected_lr = float(study_best_lr)
        selection = "best_val_rmse"
    else:
        selected_lr = float(lr_fixed)
        selection = "fixed_lr"
    selected_key = next(k for k in states_by_lr if lr_close(k, selected_lr))
    model = make_model()
    model.load_state_dict(states_by_lr[selected_key])
    model.eval()
    fit = fits_by_lr[selected_key]
    describe = describes_by_lr[selected_key]
    selected_val = float(fit["best_validation_RMSE_total"])
    for row in lr_rows:
        row["used_for_eval"] = lr_close(row["lr"], selected_lr)
    study_df = pd.DataFrame(lr_rows).sort_values("best_val_rmse")
    if verbose:
        print(
            f"Selected {tag}: lr={selected_lr:g}  best_val={selected_val:.4f}  "
            f"mode={selection}",
            flush=True,
        )
    return {
        "model": model,
        "fit": fit,
        "describe": describe,
        "selected_lr": selected_lr,
        "selected_val": selected_val,
        "selection_mode": selection,
        "study_df": study_df,
        "study_best_lr": study_best_lr,
        "study_best_val": study_best_val,
    }


def _collect_mv_predictions(
    dataset: Dataset,
    *,
    x_field: str,
    device: torch.device | str,
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    preds: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(len(dataset)):
            images, targets = dataset[i]
            views = torch.as_tensor(images[x_field], dtype=torch.float32, device=device)
            if views.ndim == 4:
                views = views.unsqueeze(0)
            tgt = torch.tensor(
                [float(targets[n]) for n in Y_FIELDS],
                dtype=torch.float32,
            ).unsqueeze(0)
            preds.append(predict_fn(views).detach().cpu())
            tgts.append(tgt)
    return torch.cat(preds), torch.cat(tgts)


def run_m9_frozen_fusion_family(cfg: M9FusionConfig) -> M9FusionFamilyResult:
    """Run (or load) Stage A + Stage B + val/test comparison for one family."""
    family = cfg.family
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device

    if family == "fourier":
        csv_comparison = "m09_comparison_fourier.csv"
        csv_lr = "m09_lr_study_fourier.csv"
        ckpt_name = "m09_frozen_fourier_fusion.pt"
        json_name = "m09_frozen_fourier_fusion.json"
        experiment_id = "m09_frozen_fourier_fusion"
        variant_order = FOURIER_VARIANT_ORDER
        display_map = FOURIER_DISPLAY
    else:
        csv_comparison = "m09_comparison_pooled.csv"
        csv_lr = "m09_lr_study_pooled.csv"
        ckpt_name = "m09_frozen_pooled_fusion.pt"
        json_name = "m09_frozen_pooled_fusion.json"
        experiment_id = "m09_frozen_pooled_fusion"
        variant_order = POOLED_VARIANT_ORDER
        display_map = POOLED_DISPLAY

    comparison_path = results_dir / csv_comparison
    lr_study_path = results_dir / csv_lr
    checkpoint_path = results_dir / ckpt_name

    block = m8_single_view_block_freeze()
    arch = block.architecture
    x_field = str(block.x_field)
    image_normalize = str(block.image_normalize)
    optical_id = str(block.optical_setup_id_reference)
    lr_stage_a = float(
        block.lr_by_role()["primary" if family == "fourier" else "negative_control"]
    )

    catalog_jobs = load_catalog_jobs(cfg.workbook_path, cfg.data_root, stl_root=cfg.stl_root)
    catalog_rows = build_catalog_rows(catalog_jobs)
    view_angles = resolve_view_angles(
        catalog_rows,
        optical_setup_id=optical_id,
        angle_stride_deg=cfg.angle_stride_deg,
    )

    have_checkpoint = checkpoint_path.is_file()
    skip_train = bool(cfg.load_existing and have_checkpoint and not cfg.retrain)

    if cfg.verbose:
        print(
            f"M9 {family}: views={view_angles}  epochs={cfg.num_epochs}  "
            f"patience={cfg.early_stop_patience}  batch={cfg.batch_size}  "
            f"lr_stage_a={lr_stage_a:g}  RUN_LR_STUDY={cfg.run_lr_study}  "
            f"SKIP_TRAIN={skip_train}",
            flush=True,
        )

    batch_xy_single = make_batch_xy_single(
        x_field=x_field, y_fields=Y_FIELDS, device=device
    )
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
            family=family,
        )

    if family == "fourier":
        backbone: nn.Module = new_frozen_single_view_expert(
            n_outputs=3, hidden=int(arch.head_hidden)
        ).to(device)

        def make_trunk() -> nn.Module:
            return new_frozen_single_view_expert(
                n_outputs=3, hidden=int(arch.head_hidden)
            ).to(device)

        def make_compact(*, packing: str) -> nn.Module:
            trunk = make_trunk()
            trunk.load_state_dict(stage_a_state)
            return CompactLatentFusionLocalizer(
                trunk,
                n_views=len(view_angles),
                fusion_hidden=int(cfg.fusion_hidden),
                freeze_encoder=True,
                packing=packing,
            ).to(device)

        def make_deepsets() -> nn.Module:
            trunk = make_trunk()
            trunk.load_state_dict(stage_a_state)
            return FrozenEncoderDeepSetsLocalizer.for_09_1_fourier(
                trunk,
                n_views=len(view_angles),
                phi_hidden=int(cfg.phi_hidden),
                rho_hidden=int(cfg.rho_hidden),
                freeze_encoder=True,
            ).to(device)

        deepsets_packing = "deepsets_fourier"
        deepsets_tag = "deepsets_fourier"
        mean_tag = "mean_pool"
        concat_tag = "ordered_concat"
        pattern_mean = FUSION_PATTERN_09_1_MEAN_POOL
        pattern_ds = FUSION_PATTERN_09_1_DEEPSETS_FOURIER
        pattern_concat = FUSION_PATTERN_09_1
        backbone_kind = "fourier"
        ckpt_stage_key = "stage_a_state"
        ckpt_ds_key = "deepsets_fourier_state"
        display_sv = "SV ref"
        display_xyz = "xyz mean"
        display_mean = "mean-pool MLP"
        display_ds = "DeepSets Fourier"
        display_concat = "ordered concat"
        id_sv, id_xyz, id_mean, id_ds, id_concat = FOURIER_VARIANT_ORDER
    else:
        backbone = new_frozen_pooled_single_view_expert(
            n_outputs=3, embed_dim=int(arch.head_hidden)
        ).to(device)

        def make_trunk() -> nn.Module:
            return new_frozen_pooled_single_view_expert(
                n_outputs=3, embed_dim=int(arch.head_hidden)
            ).to(device)

        def make_compact(*, packing: str) -> nn.Module:
            trunk = make_trunk()
            trunk.load_state_dict(stage_a_state)
            return CompactLatentFusionLocalizer(
                trunk,
                n_views=len(view_angles),
                fusion_hidden=int(cfg.fusion_hidden),
                freeze_encoder=True,
                packing=packing,
                backbone_kind="pooled",
            ).to(device)

        def make_deepsets() -> nn.Module:
            trunk = make_trunk()
            trunk.load_state_dict(stage_a_state)
            return FrozenEncoderDeepSetsLocalizer.for_09_1_no_fourier(
                trunk,
                n_views=len(view_angles),
                phi_hidden=int(cfg.phi_hidden),
                rho_hidden=int(cfg.rho_hidden),
                freeze_encoder=True,
            ).to(device)

        deepsets_packing = "deepsets_no_fourier"
        deepsets_tag = "deepsets_no_fourier"
        mean_tag = "mean_pool_pooled"
        concat_tag = "ordered_concat_pooled"
        pattern_mean = FUSION_PATTERN_09_1_MEAN_POOL_POOLED
        pattern_ds = FUSION_PATTERN_09_1_DEEPSETS_NO_FOURIER
        pattern_concat = FUSION_PATTERN_09_1_POOLED
        backbone_kind = "pooled"
        ckpt_stage_key = "stage_a_pooled_state"
        ckpt_ds_key = "deepsets_no_fourier_state"
        display_sv = "SV pooled"
        display_xyz = "xyz mean pooled"
        display_mean = "mean-pool MLP pooled"
        display_ds = "DeepSets no-Fourier"
        display_concat = "ordered concat pooled"
        id_sv, id_xyz, id_mean, id_ds, id_concat = POOLED_VARIANT_ORDER

    stage_a_state: dict[str, torch.Tensor] | None = None
    fit_a: dict[str, Any] = {
        "history": [],
        "parameter_count": int(
            sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        ),
    }

    if skip_train and checkpoint_path.is_file():
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        stage_a_state = blob.get(ckpt_stage_key)
        if stage_a_state is not None:
            backbone.load_state_dict(stage_a_state)
            if cfg.verbose:
                print(f"Loaded Stage A from {display_path(checkpoint_path)}", flush=True)
        else:
            skip_train = False

    if not skip_train:
        train_ds_sv = ConcatDataset([make_split("train", theta) for theta in view_angles])
        val_ds_sv = ConcatDataset(
            [make_split("validation", theta) for theta in view_angles]
        )
        train_loader_sv = DataLoader(
            train_ds_sv,
            batch_size=min(int(cfg.batch_size), len(train_ds_sv)),
            shuffle=True,
        )
        val_loader_sv = DataLoader(
            val_ds_sv,
            batch_size=min(int(cfg.batch_size), len(val_ds_sv)),
            shuffle=False,
        )
        if cfg.verbose:
            print(
                f"Stage A {family} train={len(train_ds_sv)}  val={len(val_ds_sv)}",
                flush=True,
            )
        fit_a = train_full_split(
            model=backbone,
            train_loader=train_loader_sv,
            batch_xy=batch_xy_single,
            device=device,
            num_epochs=int(cfg.num_epochs),
            lr=lr_stage_a,
            val_loader=val_loader_sv,
            y_fields=Y_FIELDS,
            early_stop_patience=int(cfg.early_stop_patience),
            progress_label=f"Stage A {family}",
        )
        stage_a_state = copy.deepcopy(backbone.state_dict())
    elif stage_a_state is None:
        stage_a_state = copy.deepcopy(backbone.state_dict())

    assert stage_a_state is not None

    train_ds_mv = make_split("train", list(view_angles))
    val_ds_mv = make_split("validation", list(view_angles))
    test_ds_mv = make_split("test", list(view_angles))
    train_loader_mv = DataLoader(
        train_ds_mv,
        batch_size=min(int(cfg.batch_size), len(train_ds_mv)),
        shuffle=True,
    )
    val_loader_mv = DataLoader(
        val_ds_mv,
        batch_size=min(int(cfg.batch_size), len(val_ds_mv)),
        shuffle=False,
    )

    lr_study_df = pd.DataFrame()
    selected_lrs: dict[str, float] = {}
    selection_modes: dict[str, str] = {}
    f_mean = f_concat = f_ds = None
    mean_describe = concat_describe = ds_describe = None

    if skip_train:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if comparison_path.is_file():
            comparison_df = pd.read_csv(comparison_path)
        elif blob.get("comparison") is not None:
            comparison_df = pd.DataFrame(blob["comparison"])
            comparison_df.to_csv(comparison_path, index=False)
        else:
            raise FileNotFoundError(
                f"M9 checkpoint {display_path(checkpoint_path)} has no comparison table"
            )
        if "display_label" not in comparison_df.columns:
            comparison_df["display_label"] = comparison_df["variant_id"].map(
                lambda v: display_map.get(v, v)
            )
        if lr_study_path.is_file():
            lr_study_df = pd.read_csv(lr_study_path)
        elif blob.get("lr_study") is not None:
            lr_study_df = pd.DataFrame(blob["lr_study"])
            lr_study_df.to_csv(lr_study_path, index=False)
        else:
            lr_study_df = pd.DataFrame()
        f_mean = make_compact(packing=PACKING_MEAN_POOL)
        f_concat = make_compact(packing=PACKING_ORDERED_CONCAT)
        f_ds = make_deepsets()
        if blob.get("mean_pool_state") is not None:
            f_mean.load_state_dict(blob["mean_pool_state"])
        if blob.get("model_state") is not None:
            f_concat.load_state_dict(blob["model_state"])
        if blob.get(ckpt_ds_key) is not None:
            f_ds.load_state_dict(blob[ckpt_ds_key])
        lr_map = blob.get("lr_stage_b") or {}
        selected_lrs = {
            "mean_pool": float(lr_map.get("mean_pool", cfg.lr_stage_b_mean_pool)),
            "ordered_concat": float(
                lr_map.get("ordered_concat", cfg.lr_stage_b_ordered_concat)
            ),
            "deepsets": float(
                lr_map.get(
                    "deepsets_fourier"
                    if family == "fourier"
                    else "deepsets_no_fourier",
                    cfg.lr_stage_b_deepsets,
                )
            ),
        }
        selection_modes = {
            "mean_pool": "loaded",
            "ordered_concat": "loaded",
            "deepsets": "loaded",
        }
        mean_describe = f_mean.describe()
        concat_describe = f_concat.describe()
        ds_describe = f_ds.describe()
        if cfg.verbose:
            print(
                f"SKIP_TRAIN: loaded {display_path(checkpoint_path)} "
                f"({len(comparison_df)} comparison rows)",
                flush=True,
            )
    else:
        if cfg.verbose:
            print(
                f"Stage B {family} train={len(train_ds_mv)}  val={len(val_ds_mv)}",
                flush=True,
            )
        grid = tuple(float(x) for x in cfg.lr_stage_b_grid)
        result_mean = run_fusion_lr_study(
            tag=mean_tag,
            packing_label=PACKING_MEAN_POOL,
            make_model=lambda: make_compact(packing=PACKING_MEAN_POOL),
            train_loader=train_loader_mv,
            val_loader=val_loader_mv,
            batch_xy=batch_xy_multiview,
            device=device,
            lr_fixed=float(cfg.lr_stage_b_mean_pool),
            lr_grid=grid,
            run_study=bool(cfg.run_lr_study),
            select_best_val_lr=bool(cfg.compact_select_best_val_lr),
            num_epochs=int(cfg.num_epochs),
            early_stop_patience=int(cfg.early_stop_patience),
            progress_prefix=f"09_1 {family}",
            verbose=cfg.verbose,
        )
        result_concat = run_fusion_lr_study(
            tag=concat_tag,
            packing_label=PACKING_ORDERED_CONCAT,
            make_model=lambda: make_compact(packing=PACKING_ORDERED_CONCAT),
            train_loader=train_loader_mv,
            val_loader=val_loader_mv,
            batch_xy=batch_xy_multiview,
            device=device,
            lr_fixed=float(cfg.lr_stage_b_ordered_concat),
            lr_grid=grid,
            run_study=bool(cfg.run_lr_study),
            select_best_val_lr=bool(cfg.compact_select_best_val_lr),
            num_epochs=int(cfg.num_epochs),
            early_stop_patience=int(cfg.early_stop_patience),
            progress_prefix=f"09_1 {family}",
            verbose=cfg.verbose,
        )
        result_ds = run_fusion_lr_study(
            tag=deepsets_tag,
            packing_label=deepsets_packing,
            make_model=make_deepsets,
            train_loader=train_loader_mv,
            val_loader=val_loader_mv,
            batch_xy=batch_xy_multiview,
            device=device,
            lr_fixed=float(cfg.lr_stage_b_deepsets),
            lr_grid=grid,
            run_study=bool(cfg.run_lr_study),
            select_best_val_lr=bool(cfg.deepsets_select_best_val_lr),
            num_epochs=int(cfg.num_epochs),
            early_stop_patience=int(cfg.early_stop_patience),
            progress_prefix=f"09_1 {family}",
            verbose=cfg.verbose,
        )
        f_mean = result_mean["model"]
        f_concat = result_concat["model"]
        f_ds = result_ds["model"]
        mean_describe = result_mean["describe"]
        concat_describe = result_concat["describe"]
        ds_describe = result_ds["describe"]
        selected_lrs = {
            "mean_pool": float(result_mean["selected_lr"]),
            "ordered_concat": float(result_concat["selected_lr"]),
            "deepsets": float(result_ds["selected_lr"]),
        }
        selection_modes = {
            "mean_pool": str(result_mean["selection_mode"]),
            "ordered_concat": str(result_concat["selection_mode"]),
            "deepsets": str(result_ds["selection_mode"]),
        }
        lr_study_df = pd.concat(
            [
                result_mean["study_df"],
                result_concat["study_df"],
                result_ds["study_df"],
            ],
            ignore_index=True,
        ).sort_values(["packing", "best_val_rmse"])
        lr_study_df.to_csv(lr_study_path, index=False)

        backbone.load_state_dict(stage_a_state)
        backbone.eval()
        sv_angle = 180.0 if 180.0 in view_angles else float(view_angles[len(view_angles) // 2])

        def eval_sv(split: str) -> dict[str, float]:
            ds = make_split(split, sv_angle)
            loader = DataLoader(
                ds, batch_size=min(int(cfg.batch_size), len(ds)), shuffle=False
            )
            return evaluate_split_rmse(
                model=backbone,
                loader=loader,
                batch_xy=batch_xy_single,
                y_fields=Y_FIELDS,
                prefix="train",
            )

        rows: list[dict[str, Any]] = []
        for split, mv_ds in (("validation", val_ds_mv), ("test", test_ds_mv)):
            sv = eval_sv(split)
            pred_xyz, tgt = _collect_mv_predictions(
                mv_ds,
                x_field=x_field,
                device=device,
                predict_fn=lambda v: shared_xyz_mean(backbone, v),
            )
            pred_mean, _ = _collect_mv_predictions(
                mv_ds, x_field=x_field, device=device, predict_fn=lambda v: f_mean(v)
            )
            pred_ds, _ = _collect_mv_predictions(
                mv_ds, x_field=x_field, device=device, predict_fn=lambda v: f_ds(v)
            )
            pred_concat, _ = _collect_mv_predictions(
                mv_ds, x_field=x_field, device=device, predict_fn=lambda v: f_concat(v)
            )
            m_xyz = per_axis_rmse(pred_xyz, tgt, Y_FIELDS)
            m_mean = per_axis_rmse(pred_mean, tgt, Y_FIELDS)
            m_ds = per_axis_rmse(pred_ds, tgt, Y_FIELDS)
            m_concat = per_axis_rmse(pred_concat, tgt, Y_FIELDS)
            n_backbone = int(sum(p.numel() for p in backbone.parameters()))
            rows.extend(
                [
                    {
                        "variant_id": id_sv,
                        "display_label": display_sv,
                        "fusion_pattern": f"single_view_{backbone_kind}_reference",
                        "packing": None,
                        "backbone_kind": backbone_kind,
                        "n_views": 1,
                        "split": split,
                        "encoder_frozen": False,
                        "lr_stage_b": None,
                        "lr_selection": None,
                        "RMSE_total": sv["train_RMSE_total"],
                        "RMSE_X": sv["train_RMSE_X"],
                        "RMSE_Y": sv["train_RMSE_Y"],
                        "RMSE_Z": sv["train_RMSE_Z"],
                        "learned_parameter_count": n_backbone,
                    },
                    {
                        "variant_id": id_xyz,
                        "display_label": display_xyz,
                        "fusion_pattern": "shared_xyz_mean",
                        "packing": "xyz_mean",
                        "backbone_kind": backbone_kind,
                        "n_views": len(view_angles),
                        "split": split,
                        "encoder_frozen": True,
                        "lr_stage_b": None,
                        "lr_selection": None,
                        "RMSE_total": m_xyz["train_RMSE_total"],
                        "RMSE_X": m_xyz["train_RMSE_X"],
                        "RMSE_Y": m_xyz["train_RMSE_Y"],
                        "RMSE_Z": m_xyz["train_RMSE_Z"],
                        "learned_parameter_count": n_backbone,
                    },
                    {
                        "variant_id": id_mean,
                        "display_label": display_mean,
                        "fusion_pattern": pattern_mean,
                        "packing": PACKING_MEAN_POOL,
                        "backbone_kind": backbone_kind,
                        "n_views": len(view_angles),
                        "split": split,
                        "encoder_frozen": True,
                        "lr_stage_b": selected_lrs["mean_pool"],
                        "lr_selection": selection_modes["mean_pool"],
                        "RMSE_total": m_mean["train_RMSE_total"],
                        "RMSE_X": m_mean["train_RMSE_X"],
                        "RMSE_Y": m_mean["train_RMSE_Y"],
                        "RMSE_Z": m_mean["train_RMSE_Z"],
                        "learned_parameter_count": int(f_mean.learned_parameter_count()),
                    },
                    {
                        "variant_id": id_ds,
                        "display_label": display_ds,
                        "fusion_pattern": pattern_ds,
                        "packing": deepsets_packing,
                        "backbone_kind": backbone_kind,
                        "n_views": len(view_angles),
                        "split": split,
                        "encoder_frozen": True,
                        "lr_stage_b": selected_lrs["deepsets"],
                        "lr_selection": selection_modes["deepsets"],
                        "RMSE_total": m_ds["train_RMSE_total"],
                        "RMSE_X": m_ds["train_RMSE_X"],
                        "RMSE_Y": m_ds["train_RMSE_Y"],
                        "RMSE_Z": m_ds["train_RMSE_Z"],
                        "learned_parameter_count": int(f_ds.learned_parameter_count()),
                    },
                    {
                        "variant_id": id_concat,
                        "display_label": display_concat,
                        "fusion_pattern": pattern_concat,
                        "packing": PACKING_ORDERED_CONCAT,
                        "backbone_kind": backbone_kind,
                        "n_views": len(view_angles),
                        "split": split,
                        "encoder_frozen": True,
                        "lr_stage_b": selected_lrs["ordered_concat"],
                        "lr_selection": selection_modes["ordered_concat"],
                        "RMSE_total": m_concat["train_RMSE_total"],
                        "RMSE_X": m_concat["train_RMSE_X"],
                        "RMSE_Y": m_concat["train_RMSE_Y"],
                        "RMSE_Z": m_concat["train_RMSE_Z"],
                        "learned_parameter_count": int(
                            f_concat.learned_parameter_count()
                        ),
                    },
                ]
            )

        comparison_df = pd.DataFrame(rows)
        comparison_df["variant_order"] = comparison_df["variant_id"].map(
            {v: i for i, v in enumerate(variant_order)}
        )
        comparison_df = comparison_df.sort_values(["split", "variant_order"]).drop(
            columns=["variant_order"]
        )
        comparison_df.to_csv(comparison_path, index=False)

        ds_lr_key = (
            "deepsets_fourier" if family == "fourier" else "deepsets_no_fourier"
        )
        payload = {
            "experiment_id": experiment_id,
            "family": family,
            "n_views": len(view_angles),
            "view_angles_deg": list(view_angles),
            "angle_stride_deg": float(cfg.angle_stride_deg),
            "fusion_hidden": int(cfg.fusion_hidden),
            "phi_hidden": int(cfg.phi_hidden),
            "rho_hidden": int(cfg.rho_hidden),
            "run_lr_study": bool(cfg.run_lr_study),
            "lr_stage_a": lr_stage_a,
            "lr_stage_b_mean_pool": selected_lrs["mean_pool"],
            "lr_stage_b_ordered_concat": selected_lrs["ordered_concat"],
            f"lr_stage_b_{ds_lr_key}": selected_lrs["deepsets"],
            "num_epochs": int(cfg.num_epochs),
            "early_stop_patience": int(cfg.early_stop_patience),
            "batch_size": int(cfg.batch_size),
            "lr_study": lr_study_df.to_dict(orient="records"),
            "models": {
                "mean_pool": mean_describe,
                "ordered_concat": concat_describe,
                "deepsets": ds_describe,
            },
            "stage_a_epochs": len(fit_a["history"]),
            "comparison": comparison_df.to_dict(orient="records"),
        }
        (results_dir / json_name).write_text(
            json.dumps(payload, indent=2, default=str)
        )
        torch.save(
            {
                "model_state": f_concat.state_dict(),
                "mean_pool_state": f_mean.state_dict(),
                ckpt_ds_key: f_ds.state_dict(),
                ckpt_stage_key: stage_a_state,
                "lr_stage_b": {
                    "mean_pool": selected_lrs["mean_pool"],
                    "ordered_concat": selected_lrs["ordered_concat"],
                    ds_lr_key: selected_lrs["deepsets"],
                },
                "lr_study": payload["lr_study"],
            },
            checkpoint_path,
        )
        if cfg.verbose:
            print(f"Wrote {display_path(comparison_path)}", flush=True)
            print(f"Wrote {display_path(lr_study_path)}", flush=True)
            print(f"Wrote {display_path(checkpoint_path)}", flush=True)

    return M9FusionFamilyResult(
        family=family,
        comparison_df=comparison_df,
        lr_study_df=lr_study_df,
        view_angles=view_angles,
        lr_stage_a=lr_stage_a,
        selected_lrs=selected_lrs,
        results_dir=results_dir,
        comparison_path=comparison_path,
        lr_study_path=lr_study_path,
        checkpoint_path=checkpoint_path,
        skipped_train=bool(skip_train),
        num_epochs=int(cfg.num_epochs),
        early_stop_patience=int(cfg.early_stop_patience),
        batch_size=int(cfg.batch_size),
        extra={
            "x_field": x_field,
            "image_normalize": image_normalize,
            "optical_setup_id": optical_id,
            "selection_modes": selection_modes,
        },
    )
