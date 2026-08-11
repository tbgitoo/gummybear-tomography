"""M10 hierarchical light-then-camera fusion (10_2 Fourier + pooled GAP).

Stage A warm-starts Fourier and optional pooled single-view trunks on
concatenated single-camera-angle task datasets (all lights mixed). Stage B
trains hierarchical e2e fusion heads with an optional LR sweep; evaluation
covers SV reference, flat xyz-mean, and hierarchical fusion on val/test.

Matches ``notebooks/10_2_hierarchical_light_then_camera_fusion.ipynb`` and
adds a pooled GAP negative-control family.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from gummybear.paths import display_path, repo_relative_path
from tomography_ml.gummybear_data_catalog import (
    HierarchicalCameraLightDataset,
    build_catalog_rows,
    build_illumination_joint_groups,
    count_groups_by_split,
    groups_for_split,
    load_catalog_jobs,
)
from tomography_ml.gummybear_data_catalog.task_dataset import (
    DatasetTaskSpec,
    build_task_dataset,
)
from tomography_ml.localization.architecture_capability import (
    evaluate_split_rmse,
    mse_loss,
    per_axis_rmse,
    train_full_split,
)
from tomography_ml.localization.builders import count_parameters, m8_single_view_block_freeze
from tomography_ml.localization.localize_multiview import (
    FUSION_PATTERN_10_2,
    FUSION_PATTERN_10_2_POOLED,
    M10_LIGHT_ANGLES_DEG,
    HierarchicalLightThenCameraFusionLocalizer,
    new_frozen_pooled_single_view_expert,
    new_frozen_single_view_expert,
    shared_xyz_mean,
)
from tomography_ml.studies.study_checkpoints import (
    M10_HIERARCHICAL_LIGHT_THEN_CAMERA,
    clone_state_dict,
    load_study_checkpoint,
    save_study_checkpoint,
)
from tomography_ml.training.training_helpers import make_batch_xy_single

Y_FIELDS: tuple[str, ...] = ("particle_x", "particle_y", "particle_z")
DEFAULT_LR_STAGE_B_GRID: tuple[float, ...] = (0.03, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)

CSV_COMPARISON = "m10_2_comparison.csv"
CSV_LR_STUDY = "m10_2_lr_study.csv"
JSON_NAME = "m10_2_hierarchical_light_then_camera.json"
EXPERIMENT_ID = "m10_2_hierarchical_light_then_camera"

FOURIER_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("m10_2_single_view_reference", "SV ref", "single_view_reference"),
    ("m10_2_shared_xyz_mean_joint", "xyz mean", "shared_xyz_mean"),
    (
        "m10_2_hierarchical_light_then_camera",
        "10_2 hier.",
        FUSION_PATTERN_10_2,
    ),
)
POOLED_VARIANTS: tuple[tuple[str, str, str], ...] = (
    (
        "m10_2_single_view_pooled_reference",
        "SV pooled",
        "single_view_pooled_reference",
    ),
    ("m10_2_shared_xyz_mean_pooled", "xyz mean pooled", "shared_xyz_mean"),
    (
        "m10_2_hierarchical_pooled_light_then_camera",
        "10_2 hier. pooled",
        FUSION_PATTERN_10_2_POOLED,
    ),
)


@dataclass
class M10HierarchicalConfig:
    """Hyperparameters and paths for M10 hierarchical light-then-camera fusion."""

    workbook_path: Path
    data_root: Path
    results_dir: Path
    stl_root: Path
    device: torch.device | str
    num_epochs: int = 200
    early_stop_patience: int = 40
    batch_size: int = 1
    angle_stride_deg: float = 10.0
    light_angles_deg: Sequence[float] | None = None
    min_joint_groups: int = 4
    run_lr_study: bool = True
    # LR sweeps remain illustrative; reported Stage-B models use lr_stage_b_*.
    select_best_val_lr: bool = False
    lr_stage_b_grid: Sequence[float] = DEFAULT_LR_STAGE_B_GRID
    lr_stage_b_fourier: float = 3e-4
    lr_stage_b_pooled: float = 3e-4
    include_pooled_control: bool = True
    load_existing: bool = True
    retrain: bool = False
    verbose: bool = True


@dataclass
class M10HierarchicalResult:
    """Artifacts from :func:`run_m10_hierarchical_fusion`."""

    comparison_df: pd.DataFrame
    lr_study_df: pd.DataFrame
    view_angles: tuple[float, ...]
    light_angles: tuple[float, ...]
    selected_lrs: dict[str, Any]
    results_dir: Path
    comparison_path: Path
    checkpoint_path: Path
    lr_study_path: Path
    skipped_train: bool
    extra: dict[str, Any] = field(default_factory=dict)


def _resolve_view_angles(
    joint_groups: Sequence[dict[str, Any]],
    light_angles: Sequence[float],
    angle_stride_deg: float,
) -> tuple[float, ...]:
    """Subsample the catalog camera orbit onto a regular stride."""
    if not joint_groups:
        raise ValueError("joint_groups must be non-empty to resolve view angles")
    first_light = float(light_angles[0])
    orbit = tuple(
        float(a)
        for a in joint_groups[0]["rows_by_light"][first_light].angles_deg
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
        raise ValueError(
            f"Need ≥2 camera views; got {views} from orbit={orbit} "
            f"(stride={angle_stride_deg})"
        )
    return views


def _batch_from_indices(
    ds: HierarchicalCameraLightDataset,
    indices: Sequence[int],
    *,
    x_field: str,
    y_fields: Sequence[str],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    views_list: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    for i in indices:
        x, targets = ds[int(i)]
        views_list.append(torch.as_tensor(x[x_field], dtype=torch.float32))
        tgts.append(torch.tensor([float(targets[n]) for n in y_fields]))
    batch = len(views_list)
    cams = (
        torch.tensor(ds.camera_angles_deg, dtype=torch.float32)
        .unsqueeze(0)
        .expand(batch, -1)
    )
    lights = (
        torch.tensor(ds.light_angles_deg, dtype=torch.float32)
        .unsqueeze(0)
        .expand(batch, -1)
    )
    return (
        torch.stack(views_list, dim=0).to(device=device),
        torch.stack(tgts, dim=0).to(device=device, dtype=torch.float32),
        cams.to(device=device),
        lights.to(device=device),
    )


def _train_hierarchical_e2e(
    model: nn.Module,
    train_ds: HierarchicalCameraLightDataset,
    val_ds: HierarchicalCameraLightDataset,
    *,
    x_field: str,
    y_fields: Sequence[str],
    device: torch.device | str,
    lr: float,
    num_epochs: int,
    early_stop_patience: int,
    batch_size: int,
    progress_label: str = "10_2",
    verbose: bool = True,
) -> dict[str, Any]:
    """Adam + MSE e2e loop with early stop on validation RMSE (notebook ``train_e2e``)."""
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stall = 0
    history: list[dict[str, Any]] = []
    n_train = len(train_ds)
    bs = max(1, int(batch_size))

    for epoch in range(int(num_epochs)):
        model.train()
        order = torch.randperm(n_train)
        total = 0.0
        count = 0
        for start in range(0, n_train, bs):
            idx = order[start : start + bs].tolist()
            views, y, cams, lights = _batch_from_indices(
                train_ds,
                idx,
                x_field=x_field,
                y_fields=y_fields,
                device=device,
            )
            pred = model(
                views,
                camera_angles_deg=cams,
                light_angles_deg=lights,
            )
            loss = mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * views.shape[0]
            count += int(views.shape[0])
        train_loss = total / max(count, 1)

        model.eval()
        preds: list[torch.Tensor] = []
        tgts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(val_ds), bs):
                idx = list(range(start, min(start + bs, len(val_ds))))
                views, y, cams, lights = _batch_from_indices(
                    val_ds,
                    idx,
                    x_field=x_field,
                    y_fields=y_fields,
                    device=device,
                )
                pred = model(
                    views,
                    camera_angles_deg=cams,
                    light_angles_deg=lights,
                )
                preds.append(pred.cpu())
                tgts.append(y.cpu())
        metrics = per_axis_rmse(torch.cat(preds), torch.cat(tgts), y_fields)
        val_rmse = float(metrics["train_RMSE_total"])
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        improved = val_rmse < best_val
        if improved:
            best_val = val_rmse
            best_state = clone_state_dict(model)
            stall = 0
        else:
            stall += 1
        if verbose:
            marker = "*" if improved else " "
            print(
                f"  [{progress_label}] epoch {epoch + 1}/{num_epochs}  "
                f"train_loss={train_loss:.4f}  val_RMSE={val_rmse:.4f}  "
                f"best={best_val:.4f}{marker}  "
                f"stall={stall}/{early_stop_patience}",
                flush=True,
            )
        if stall >= int(early_stop_patience):
            if verbose:
                print(
                    f"  early stop at epoch {epoch + 1} "
                    f"(no val improvement for {early_stop_patience} epochs)",
                    flush=True,
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_rmse": best_val}


def _eval_hierarchical(
    ds: HierarchicalCameraLightDataset,
    predict_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    *,
    x_field: str,
    y_fields: Sequence[str],
    device: torch.device | str,
    batch_size: int,
) -> dict[str, float]:
    preds: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    bs = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(ds), bs):
            idx = list(range(start, min(start + bs, len(ds))))
            views, y, cams, lights = _batch_from_indices(
                ds,
                idx,
                x_field=x_field,
                y_fields=y_fields,
                device=device,
            )
            preds.append(predict_fn(views, cams, lights).cpu())
            tgts.append(y.cpu())
    return per_axis_rmse(torch.cat(preds), torch.cat(tgts), y_fields)


def _stage_b_family(
    *,
    backbone_kind: str,
    make_model: Callable[[], nn.Module],
    train_ds: HierarchicalCameraLightDataset,
    val_ds: HierarchicalCameraLightDataset,
    x_field: str,
    device: torch.device | str,
    lr_fixed: float,
    run_study: bool,
    select_best_val_lr: bool,
    lr_grid: Sequence[float],
    num_epochs: int,
    early_stop_patience: int,
    batch_size: int,
    verbose: bool,
) -> dict[str, Any]:
    """LR sweep or single Stage B run for one backbone family."""
    if run_study:
        candidates = tuple(float(x) for x in lr_grid)
        if not any(abs(float(x) - float(lr_fixed)) < 1e-15 for x in candidates):
            candidates = candidates + (float(lr_fixed),)
    else:
        candidates = (float(lr_fixed),)

    lr_rows: list[dict[str, Any]] = []
    best_lr = float(lr_fixed)
    best_val_global = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    best_fit: dict[str, Any] | None = None
    best_describe: dict[str, Any] | None = None
    fixed_model_state: dict[str, torch.Tensor] | None = None
    fixed_fit: dict[str, Any] | None = None
    fixed_describe: dict[str, Any] | None = None

    for i, lr_b in enumerate(candidates):
        if verbose:
            print(
                f"\n=== Stage B 10_2 {backbone_kind}  lr={lr_b:g}  "
                f"({i + 1}/{len(candidates)}) ===",
                flush=True,
            )
        model_b = make_model()
        fit_b = _train_hierarchical_e2e(
            model_b,
            train_ds,
            val_ds,
            x_field=x_field,
            y_fields=Y_FIELDS,
            device=device,
            lr=lr_b,
            num_epochs=num_epochs,
            early_stop_patience=early_stop_patience,
            batch_size=batch_size,
            progress_label=f"10_2 {backbone_kind} lr={lr_b:g}",
            verbose=verbose,
        )
        describe_b = (
            model_b.describe()
            if hasattr(model_b, "describe")
            else {"learned_parameter_count": count_parameters(model_b)}
        )
        state_b = clone_state_dict(model_b)
        final_train = (
            float(fit_b["history"][-1]["train_loss"])
            if fit_b["history"]
            else float("nan")
        )
        lr_rows.append(
            {
                "backbone_kind": backbone_kind,
                "lr": lr_b,
                "best_val_rmse": fit_b["best_val_rmse"],
                "epochs_ran": len(fit_b["history"]),
                "final_train_loss": final_train,
                "converged_hint": bool(
                    fit_b["best_val_rmse"] < 2.0 and final_train < 10.0
                ),
            }
        )
        if verbose:
            print(
                f"  → {backbone_kind} lr={lr_b:g} "
                f"best_val={fit_b['best_val_rmse']:.4f} "
                f"epochs={len(fit_b['history'])} "
                f"final_train_loss={final_train:.4f}",
                flush=True,
            )
        if abs(lr_b - float(lr_fixed)) < 1e-15 or (
            not run_study and fixed_model_state is None
        ):
            fixed_model_state = state_b
            fixed_fit = fit_b
            fixed_describe = describe_b
        if fit_b["best_val_rmse"] < best_val_global:
            best_val_global = float(fit_b["best_val_rmse"])
            best_lr = float(lr_b)
            best_model_state = state_b
            best_fit = fit_b
            best_describe = describe_b

    if select_best_val_lr or not run_study:
        selected_lr = float(best_lr)
        selected_state = best_model_state
        selected_fit = best_fit
        selected_describe = best_describe
    else:
        selected_lr = float(lr_fixed)
        selected_state = fixed_model_state or best_model_state
        selected_fit = fixed_fit or best_fit
        selected_describe = fixed_describe or best_describe

    assert selected_state is not None and selected_fit is not None
    model_out = make_model()
    model_out.load_state_dict(selected_state)
    model_out.eval()
    study_df = pd.DataFrame(lr_rows)
    if not study_df.empty:
        study_df = study_df.sort_values("best_val_rmse").reset_index(drop=True)
    return {
        "model": model_out,
        "selected_lr": selected_lr,
        "fit": selected_fit,
        "describe": selected_describe,
        "study_df": study_df,
        "state": selected_state,
    }


def _load_existing_result(
    cfg: M10HierarchicalConfig,
    checkpoint_path: Path,
    *,
    light_angles: tuple[float, ...],
) -> M10HierarchicalResult:
    results_dir = Path(cfg.results_dir)
    comparison_path = results_dir / CSV_COMPARISON
    lr_study_path = results_dir / CSV_LR_STUDY
    blob = load_study_checkpoint(checkpoint_path)

    if comparison_path.is_file():
        comparison_df = pd.read_csv(comparison_path)
    elif blob.get("comparison") is not None:
        comparison_df = pd.DataFrame(blob["comparison"])
        comparison_df.to_csv(comparison_path, index=False)
    else:
        raise FileNotFoundError(
            f"M10 hierarchical checkpoint {display_path(checkpoint_path)} "
            "has no comparison table"
        )

    if lr_study_path.is_file():
        lr_study_df = pd.read_csv(lr_study_path)
    elif blob.get("lr_study") is not None:
        lr_study_df = pd.DataFrame(blob["lr_study"])
        lr_study_df.to_csv(lr_study_path, index=False)
    else:
        lr_study_df = pd.DataFrame()

    selected = blob.get("selected_lrs")
    if not isinstance(selected, dict):
        selected = {
            "fourier": blob.get("lr_stage_b"),
            "pooled": blob.get("lr_stage_b_pooled"),
        }

    view_angles = tuple(
        float(a) for a in (blob.get("view_angles_deg") or [])
    )
    lights = tuple(
        float(a)
        for a in (blob.get("light_angles_deg") or list(light_angles))
    )

    if cfg.verbose:
        print(
            f"SKIP_TRAIN: loaded {display_path(checkpoint_path)} "
            f"({len(comparison_df)} comparison rows)",
            flush=True,
        )

    return M10HierarchicalResult(
        comparison_df=comparison_df,
        lr_study_df=lr_study_df,
        view_angles=view_angles,
        light_angles=lights,
        selected_lrs=dict(selected),
        results_dir=results_dir,
        comparison_path=comparison_path,
        checkpoint_path=checkpoint_path,
        lr_study_path=lr_study_path,
        skipped_train=True,
        extra={
            "experiment_id": EXPERIMENT_ID,
            "loaded_from_checkpoint": True,
        },
    )


def run_m10_hierarchical_fusion(cfg: M10HierarchicalConfig) -> M10HierarchicalResult:
    """Run (or load) M10 hierarchical light-then-camera fusion (Fourier + pooled)."""
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device
    checkpoint_path = results_dir / M10_HIERARCHICAL_LIGHT_THEN_CAMERA
    comparison_path = results_dir / CSV_COMPARISON
    lr_study_path = results_dir / CSV_LR_STUDY
    json_path = results_dir / JSON_NAME

    light_angles = tuple(
        float(a)
        for a in (
            cfg.light_angles_deg
            if cfg.light_angles_deg is not None
            else M10_LIGHT_ANGLES_DEG
        )
    )
    if len(light_angles) < 2:
        raise ValueError(f"Need ≥2 light angles; got {light_angles}")

    skip_train = bool(
        cfg.load_existing and checkpoint_path.is_file() and not cfg.retrain
    )

    block = m8_single_view_block_freeze()
    arch = block.architecture
    x_field = str(block.x_field)
    image_normalize = str(block.image_normalize)
    roles = block.lr_by_role()
    lr_stage_a_fourier = float(roles["primary"])
    lr_stage_a_pooled = float(roles.get("negative_control", roles["primary"]))
    sv_batch = int(getattr(block, "batch_size", 16) or 16)

    if cfg.verbose:
        print(
            f"M10 hierarchical 10_2: lights={light_angles}  "
            f"stride={cfg.angle_stride_deg}  epochs={cfg.num_epochs}  "
            f"patience={cfg.early_stop_patience}  batch={cfg.batch_size}  "
            f"RUN_LR_STUDY={cfg.run_lr_study}  "
            f"pooled={cfg.include_pooled_control}  SKIP_TRAIN={skip_train}",
            flush=True,
        )
        print(f"results={repo_relative_path(results_dir)}", flush=True)

    if skip_train:
        return _load_existing_result(cfg, checkpoint_path, light_angles=light_angles)

    catalog_jobs = load_catalog_jobs(
        cfg.workbook_path, cfg.data_root, stl_root=cfg.stl_root
    )
    catalog_rows = build_catalog_rows(catalog_jobs)
    joint_groups = build_illumination_joint_groups(
        catalog_rows,
        light_angles_deg=light_angles,
        min_groups=int(cfg.min_joint_groups),
    )
    view_angles = _resolve_view_angles(
        joint_groups, light_angles, float(cfg.angle_stride_deg)
    )
    n_cam = len(view_angles)
    n_light = len(light_angles)

    if cfg.verbose:
        print(
            f"joint groups={len(joint_groups)}  "
            f"splits={count_groups_by_split(joint_groups)}",
            flush=True,
        )
        print(
            f"camera VIEW_ANGLES n={n_cam} stride={cfg.angle_stride_deg}: "
            f"{view_angles}",
            flush=True,
        )
        print(f"hierarchical unit = {n_cam} cams × {n_light} lights", flush=True)

    train_ds = HierarchicalCameraLightDataset(
        groups_for_split(joint_groups, "train"),
        x_field=x_field,
        y_fields=Y_FIELDS,
        camera_angles_deg=view_angles,
        light_angles_deg=light_angles,
        image_normalize=image_normalize,
        task_name="m10_2_hierarchical_camera_light",
    )
    val_ds = HierarchicalCameraLightDataset(
        groups_for_split(joint_groups, "validation"),
        x_field=x_field,
        y_fields=Y_FIELDS,
        camera_angles_deg=view_angles,
        light_angles_deg=light_angles,
        image_normalize=image_normalize,
        task_name="m10_2_hierarchical_camera_light",
    )
    test_ds = HierarchicalCameraLightDataset(
        groups_for_split(joint_groups, "test"),
        x_field=x_field,
        y_fields=Y_FIELDS,
        camera_angles_deg=view_angles,
        light_angles_deg=light_angles,
        image_normalize=image_normalize,
        task_name="m10_2_hierarchical_camera_light",
    )
    if cfg.verbose:
        print(
            f"hierarchical datasets train/val/test = "
            f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}",
            flush=True,
        )

    def make_sv_dataset(split: str, keep_angles_deg: float | Sequence[float]) -> Dataset:
        # All lights mixed (no optical_setup_id filter) — matches notebook 10_2.
        task = DatasetTaskSpec(
            name=f"m10_2_sv_{split}_{keep_angles_deg}",
            row_filter={"field_status": "complete", "split": split},
            x_fields=(x_field,),
            y_fields=Y_FIELDS,
            keep_angles_deg=keep_angles_deg,
            image_normalize=image_normalize,
        )
        return build_task_dataset(catalog_rows, task)

    batch_xy_single = make_batch_xy_single(
        x_field=x_field, y_fields=Y_FIELDS, device=device
    )
    train_ds_sv = ConcatDataset(
        [make_sv_dataset("train", theta) for theta in view_angles]
    )
    val_ds_sv = ConcatDataset(
        [make_sv_dataset("validation", theta) for theta in view_angles]
    )
    train_loader_sv = DataLoader(
        train_ds_sv,
        batch_size=min(sv_batch, len(train_ds_sv)),
        shuffle=True,
    )
    val_loader_sv = DataLoader(
        val_ds_sv,
        batch_size=min(sv_batch, len(val_ds_sv)),
        shuffle=False,
    )

    # --- Stage A Fourier ---
    backbone = new_frozen_single_view_expert(
        n_outputs=3, hidden=int(arch.head_hidden)
    ).to(device)
    if cfg.verbose:
        print(
            f"=== Stage A Fourier  lr={lr_stage_a_fourier:g}  "
            f"n_train={len(train_ds_sv)}  n_val={len(val_ds_sv)}  "
            f"epochs≤{cfg.num_epochs} ===",
            flush=True,
        )
    fit_a = train_full_split(
        model=backbone,
        train_loader=train_loader_sv,
        batch_xy=batch_xy_single,
        device=device,
        num_epochs=int(cfg.num_epochs),
        lr=lr_stage_a_fourier,
        val_loader=val_loader_sv,
        y_fields=Y_FIELDS,
        early_stop_patience=int(cfg.early_stop_patience),
        progress_label="Stage A Fourier",
    )
    stage_a_fourier = copy.deepcopy(backbone.state_dict())
    n_backbone_f = int(count_parameters(backbone))
    if cfg.verbose:
        print(f"Stage A Fourier epochs={len(fit_a['history'])}", flush=True)

    def make_10_2_fourier() -> nn.Module:
        trunk = new_frozen_single_view_expert(
            n_outputs=3, hidden=int(arch.head_hidden)
        ).to(device)
        trunk.load_state_dict(stage_a_fourier)
        return HierarchicalLightThenCameraFusionLocalizer.for_10_2(
            trunk,
            n_cameras=n_cam,
            n_lights=n_light,
            camera_angles_deg=view_angles,
            light_angles_deg=light_angles,
            flat_layout="light_major",
        ).to(device)

    result_f = _stage_b_family(
        backbone_kind="fourier",
        make_model=make_10_2_fourier,
        train_ds=train_ds,
        val_ds=val_ds,
        x_field=x_field,
        device=device,
        lr_fixed=float(cfg.lr_stage_b_fourier),
        run_study=bool(cfg.run_lr_study),
        select_best_val_lr=bool(cfg.select_best_val_lr),
        lr_grid=tuple(float(x) for x in cfg.lr_stage_b_grid),
        num_epochs=int(cfg.num_epochs),
        early_stop_patience=int(cfg.early_stop_patience),
        batch_size=int(cfg.batch_size),
        verbose=bool(cfg.verbose),
    )
    model_fourier = result_f["model"]
    selected_lr_f = float(result_f["selected_lr"])
    describe_f = result_f["describe"] or {}
    n_params_f = int(
        describe_f.get("learned_parameter_count")
        or model_fourier.learned_parameter_count()
    )

    stage_a_pooled: dict[str, torch.Tensor] | None = None
    backbone_p: nn.Module | None = None
    model_pooled: nn.Module | None = None
    selected_lr_p: float | None = None
    describe_p: dict[str, Any] = {}
    n_backbone_p = 0
    n_params_p = 0
    lr_study_parts = [result_f["study_df"]]

    if cfg.include_pooled_control:
        backbone_p = new_frozen_pooled_single_view_expert(
            n_outputs=3, embed_dim=int(arch.head_hidden)
        ).to(device)
        if cfg.verbose:
            print(
                f"=== Stage A pooled  lr={lr_stage_a_pooled:g}  "
                f"n_train={len(train_ds_sv)}  n_val={len(val_ds_sv)}  "
                f"epochs≤{cfg.num_epochs} ===",
                flush=True,
            )
        fit_a_p = train_full_split(
            model=backbone_p,
            train_loader=train_loader_sv,
            batch_xy=batch_xy_single,
            device=device,
            num_epochs=int(cfg.num_epochs),
            lr=lr_stage_a_pooled,
            val_loader=val_loader_sv,
            y_fields=Y_FIELDS,
            early_stop_patience=int(cfg.early_stop_patience),
            progress_label="Stage A pooled",
        )
        stage_a_pooled = copy.deepcopy(backbone_p.state_dict())
        n_backbone_p = int(count_parameters(backbone_p))
        if cfg.verbose:
            print(
                f"Stage A pooled epochs={len(fit_a_p['history'])}",
                flush=True,
            )

        def make_10_2_pooled() -> nn.Module:
            trunk = new_frozen_pooled_single_view_expert(
                n_outputs=3, embed_dim=int(arch.head_hidden)
            ).to(device)
            assert stage_a_pooled is not None
            trunk.load_state_dict(stage_a_pooled)
            return HierarchicalLightThenCameraFusionLocalizer.for_10_2_pooled(
                trunk,
                n_cameras=n_cam,
                n_lights=n_light,
                camera_angles_deg=view_angles,
                light_angles_deg=light_angles,
                flat_layout="light_major",
            ).to(device)

        result_p = _stage_b_family(
            backbone_kind="pooled",
            make_model=make_10_2_pooled,
            train_ds=train_ds,
            val_ds=val_ds,
            x_field=x_field,
            device=device,
            lr_fixed=float(cfg.lr_stage_b_pooled),
            run_study=bool(cfg.run_lr_study),
            select_best_val_lr=bool(cfg.select_best_val_lr),
            lr_grid=tuple(float(x) for x in cfg.lr_stage_b_grid),
            num_epochs=int(cfg.num_epochs),
            early_stop_patience=int(cfg.early_stop_patience),
            batch_size=int(cfg.batch_size),
            verbose=bool(cfg.verbose),
        )
        model_pooled = result_p["model"]
        selected_lr_p = float(result_p["selected_lr"])
        describe_p = result_p["describe"] or {}
        n_params_p = int(
            describe_p.get("learned_parameter_count")
            or model_pooled.learned_parameter_count()
        )
        lr_study_parts.append(result_p["study_df"])
        backbone_p.load_state_dict(stage_a_pooled)
        backbone_p.eval()

    backbone.load_state_dict(stage_a_fourier)
    backbone.eval()

    sv_angle = (
        180.0 if 180.0 in view_angles else float(view_angles[len(view_angles) // 2])
    )

    def eval_sv(split: str, model_sv: nn.Module) -> dict[str, float]:
        ds = make_sv_dataset(split, sv_angle)
        loader = DataLoader(
            ds, batch_size=min(sv_batch, max(len(ds), 1)), shuffle=False
        )
        return evaluate_split_rmse(
            model=model_sv,
            loader=loader,
            batch_xy=batch_xy_single,
            y_fields=Y_FIELDS,
            prefix="train",
        )

    def mean_flat_for(
        sv_model: nn.Module,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        def _fn(views: torch.Tensor, _cams: torch.Tensor, _lights: torch.Tensor):
            b, n_i, n_v, c, h, w = views.shape
            return shared_xyz_mean(
                sv_model, views.reshape(b, n_i * n_v, c, h, w)
            )

        return _fn

    comparison_rows: list[dict[str, Any]] = []
    for split, ds in (("validation", val_ds), ("test", test_ds)):
        # Fourier family
        m_sv_f = eval_sv(split, backbone)
        m_mean_f = _eval_hierarchical(
            ds,
            mean_flat_for(backbone),
            x_field=x_field,
            y_fields=Y_FIELDS,
            device=device,
            batch_size=int(cfg.batch_size),
        )
        m_hier_f = _eval_hierarchical(
            ds,
            lambda v, c, l: model_fourier(
                v, camera_angles_deg=c, light_angles_deg=l
            ),
            x_field=x_field,
            y_fields=Y_FIELDS,
            device=device,
            batch_size=int(cfg.batch_size),
        )
        metrics_f = (m_sv_f, m_mean_f, m_hier_f)
        for (variant_id, display_label, pattern), m in zip(
            FOURIER_VARIANTS, metrics_f
        ):
            is_sv = "single_view" in variant_id
            comparison_rows.append(
                {
                    "variant_id": variant_id,
                    "display_label": display_label,
                    "backbone_kind": "fourier",
                    "fusion_pattern": pattern,
                    "split": split,
                    "n_cameras": 1 if is_sv else n_cam,
                    "n_lights": 1 if is_sv else n_light,
                    "lr_stage_b": selected_lr_f,
                    "RMSE_total": m["train_RMSE_total"],
                    "RMSE_X": m["train_RMSE_X"],
                    "RMSE_Y": m["train_RMSE_Y"],
                    "RMSE_Z": m["train_RMSE_Z"],
                    "learned_parameter_count": (
                        n_backbone_f if is_sv or "xyz_mean" in variant_id else n_params_f
                    ),
                }
            )

        if (
            cfg.include_pooled_control
            and model_pooled is not None
            and backbone_p is not None
        ):
            m_sv_p = eval_sv(split, backbone_p)
            m_mean_p = _eval_hierarchical(
                ds,
                mean_flat_for(backbone_p),
                x_field=x_field,
                y_fields=Y_FIELDS,
                device=device,
                batch_size=int(cfg.batch_size),
            )
            pooled_model = model_pooled
            m_hier_p = _eval_hierarchical(
                ds,
                lambda v, c, l: pooled_model(
                    v, camera_angles_deg=c, light_angles_deg=l
                ),
                x_field=x_field,
                y_fields=Y_FIELDS,
                device=device,
                batch_size=int(cfg.batch_size),
            )
            metrics_p = (m_sv_p, m_mean_p, m_hier_p)
            for (variant_id, display_label, pattern), m in zip(
                POOLED_VARIANTS, metrics_p
            ):
                is_sv = "single_view" in variant_id
                comparison_rows.append(
                    {
                        "variant_id": variant_id,
                        "display_label": display_label,
                        "backbone_kind": "pooled",
                        "fusion_pattern": pattern,
                        "split": split,
                        "n_cameras": 1 if is_sv else n_cam,
                        "n_lights": 1 if is_sv else n_light,
                        "lr_stage_b": selected_lr_p,
                        "RMSE_total": m["train_RMSE_total"],
                        "RMSE_X": m["train_RMSE_X"],
                        "RMSE_Y": m["train_RMSE_Y"],
                        "RMSE_Z": m["train_RMSE_Z"],
                        "learned_parameter_count": (
                            n_backbone_p
                            if is_sv or "xyz_mean" in variant_id
                            else n_params_p
                        ),
                    }
                )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(comparison_path, index=False)

    lr_study_df = pd.concat(
        [df for df in lr_study_parts if df is not None and not df.empty],
        ignore_index=True,
    )
    if not lr_study_df.empty:
        lr_study_df = lr_study_df.sort_values(
            ["backbone_kind", "best_val_rmse"]
        ).reset_index(drop=True)
    lr_study_df.to_csv(lr_study_path, index=False)

    selected_lrs: dict[str, Any] = {
        "fourier": selected_lr_f,
        "pooled": selected_lr_p,
    }

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "sample_unit": "joint_particle_hierarchical_camera_light",
        "workbook": str(repo_relative_path(cfg.workbook_path)),
        "data_root": str(repo_relative_path(cfg.data_root)),
        "run_lr_study": bool(cfg.run_lr_study),
        "select_best_val_lr": bool(cfg.select_best_val_lr),
        "lr_stage_a_fourier": lr_stage_a_fourier,
        "lr_stage_a_pooled": lr_stage_a_pooled,
        "selected_lrs": selected_lrs,
        "lr_stage_b_grid": list(cfg.lr_stage_b_grid),
        "lr_study": lr_study_df.to_dict(orient="records"),
        "angle_stride_deg": float(cfg.angle_stride_deg),
        "n_cameras": n_cam,
        "n_lights": n_light,
        "view_angles_deg": list(view_angles),
        "light_angles_deg": list(light_angles),
        "include_pooled_control": bool(cfg.include_pooled_control),
        "model_fourier": describe_f,
        "model_pooled": describe_p,
        "comparison": comparison_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    save_study_checkpoint(
        checkpoint_path,
        {
            "stage_a_fourier": stage_a_fourier,
            "stage_a_pooled": stage_a_pooled,
            "model_fourier": clone_state_dict(model_fourier),
            "model_pooled": (
                None if model_pooled is None else clone_state_dict(model_pooled)
            ),
            "comparison": comparison_df.to_dict(orient="records"),
            "lr_study": lr_study_df.to_dict(orient="records"),
            "selected_lrs": selected_lrs,
            "view_angles_deg": list(view_angles),
            "light_angles_deg": list(light_angles),
            "describe_fourier": describe_f,
            "describe_pooled": describe_p,
        },
    )

    if cfg.verbose:
        print(f"Wrote {repo_relative_path(comparison_path)}", flush=True)
        print(f"Wrote {repo_relative_path(lr_study_path)}", flush=True)
        print(f"Wrote {repo_relative_path(json_path)}", flush=True)
        print(f"Wrote {repo_relative_path(checkpoint_path)}", flush=True)
        print(comparison_df.to_string(index=False), flush=True)

    return M10HierarchicalResult(
        comparison_df=comparison_df,
        lr_study_df=lr_study_df,
        view_angles=view_angles,
        light_angles=light_angles,
        selected_lrs=selected_lrs,
        results_dir=results_dir,
        comparison_path=comparison_path,
        checkpoint_path=checkpoint_path,
        lr_study_path=lr_study_path,
        skipped_train=False,
        extra={
            "experiment_id": EXPERIMENT_ID,
            "json_path": json_path,
            "sv_angle_deg": sv_angle,
            "describe_fourier": describe_f,
            "describe_pooled": describe_p,
        },
    )


__all__ = [
    "DEFAULT_LR_STAGE_B_GRID",
    "M10HierarchicalConfig",
    "M10HierarchicalResult",
    "run_m10_hierarchical_fusion",
]
