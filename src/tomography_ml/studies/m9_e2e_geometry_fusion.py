"""M9 end-to-end geometry-aware fusion (09_2A Fourier + 09_2B pooled).

Stage A warm-starts a single-view trunk; Stage B jointly trains trunk +
camera sin/cos fusion (compact 09_2 and large 09_3 heads).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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
    FUSION_PATTERN_09_2,
    FUSION_PATTERN_09_2_POOLED,
    FUSION_PATTERN_09_3,
    FUSION_PATTERN_09_3_POOLED,
    M9_2_FUSION_DEPTH,
    M9_2_FUSION_HIDDEN,
    M9_3_FUSION_DEPTH,
    M9_3_FUSION_HIDDEN,
    GeometryAwareFourierFusionLocalizer,
    new_frozen_pooled_single_view_expert,
    new_frozen_single_view_expert,
    shared_xyz_mean,
)
from tomography_ml.studies.m9_frozen_fusion import (
    _collect_mv_predictions,
    resolve_view_angles,
)
from tomography_ml.studies.study_checkpoints import (
    M09_E2E_FOURIER_GEOMETRY_FUSION,
    M09_E2E_POOLED_GEOMETRY_FUSION,
    clone_state_dict,
    save_study_checkpoint,
)
from tomography_ml.training.training_helpers import (
    make_batch_xy_multiview,
    make_batch_xy_single,
)

Family = Literal["fourier", "pooled"]

Y_FIELDS: tuple[str, ...] = ("particle_x", "particle_y", "particle_z")

FOURIER_VARIANT_ORDER: tuple[str, ...] = (
    "f2a_single_view_reference",
    "f2a_shared_xyz_mean_control",
    "m09_2_e2e_fourier_geometry_fusion",
    "m09_3_e2e_fourier_geometry_large_fusion",
)
POOLED_VARIANT_ORDER: tuple[str, ...] = (
    "f2b_single_view_pooled_reference",
    "f2b_shared_xyz_mean_pooled_control",
    "m09_2_e2e_pooled_geometry_fusion",
    "m09_3_e2e_pooled_geometry_large_fusion",
)

FOURIER_DISPLAY: dict[str, str] = {
    "f2a_single_view_reference": "SV ref",
    "f2a_shared_xyz_mean_control": "xyz mean",
    "m09_2_e2e_fourier_geometry_fusion": "09_2 compact",
    "m09_3_e2e_fourier_geometry_large_fusion": "09_3 large",
}
POOLED_DISPLAY: dict[str, str] = {
    "f2b_single_view_pooled_reference": "SV pooled",
    "f2b_shared_xyz_mean_pooled_control": "xyz mean pooled",
    "m09_2_e2e_pooled_geometry_fusion": "09_2 compact pooled",
    "m09_3_e2e_pooled_geometry_large_fusion": "09_3 large pooled",
}


@dataclass
class M9E2EConfig:
    """Hyperparameters and paths for one M9 e2e geometry-fusion family run."""

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
    load_existing: bool = True
    retrain: bool = False
    fusion_hidden: int = M9_2_FUSION_HIDDEN
    fusion_depth: int = M9_2_FUSION_DEPTH
    fusion_hidden_large: int = M9_3_FUSION_HIDDEN
    fusion_depth_large: int = M9_3_FUSION_DEPTH
    verbose: bool = True


@dataclass
class M9E2EFamilyResult:
    """Artifacts from :func:`run_m9_e2e_geometry_fusion_family`."""

    family: Family
    comparison_df: pd.DataFrame
    view_angles: tuple[float, ...]
    lr_stage_a: float
    results_dir: Path
    comparison_path: Path
    checkpoint_path: Path
    skipped_train: bool
    num_epochs: int
    early_stop_patience: int
    batch_size: int
    compact_describe: dict[str, Any] = field(default_factory=dict)
    large_describe: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _make_split_dataset(
    catalog_rows: Sequence[Any],
    *,
    split: str,
    keep_angles_deg: float | Sequence[float],
    x_field: str,
    image_normalize: str,
    optical_setup_id: str,
    family: Family,
) -> Dataset:
    task = DatasetTaskSpec(
        name=f"m9_e2e_{family}_{split}_{keep_angles_deg}",
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


def run_m9_e2e_geometry_fusion_family(cfg: M9E2EConfig) -> M9E2EFamilyResult:
    """Run (or load) Stage A + e2e Stage B + val/test comparison for one family."""
    family = cfg.family
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device

    if family == "fourier":
        csv_comparison = "m09_e2e_comparison_fourier.csv"
        ckpt_name = M09_E2E_FOURIER_GEOMETRY_FUSION
        json_name = "m09_e2e_fourier_geometry_fusion.json"
        experiment_id = "m09_e2e_fourier_geometry_fusion"
        variant_order = FOURIER_VARIANT_ORDER
        display_map = FOURIER_DISPLAY
        ckpt_stage_key = "stage_a_state"
        backbone_kind = "fourier"
        id_sv, id_xyz, id_compact, id_large = FOURIER_VARIANT_ORDER
        display_sv, display_xyz = "SV ref", "xyz mean"
        display_compact, display_large = "09_2 compact", "09_3 large"
        pattern_compact = FUSION_PATTERN_09_2
        pattern_large = FUSION_PATTERN_09_3
        pattern_sv = "single_view_reference"
    else:
        csv_comparison = "m09_e2e_comparison_pooled.csv"
        ckpt_name = M09_E2E_POOLED_GEOMETRY_FUSION
        json_name = "m09_e2e_pooled_geometry_fusion.json"
        experiment_id = "m09_e2e_pooled_geometry_fusion"
        variant_order = POOLED_VARIANT_ORDER
        display_map = POOLED_DISPLAY
        ckpt_stage_key = "stage_a_pooled_state"
        backbone_kind = "pooled"
        id_sv, id_xyz, id_compact, id_large = POOLED_VARIANT_ORDER
        display_sv, display_xyz = "SV pooled", "xyz mean pooled"
        display_compact, display_large = "09_2 compact pooled", "09_3 large pooled"
        pattern_compact = FUSION_PATTERN_09_2_POOLED
        pattern_large = FUSION_PATTERN_09_3_POOLED
        pattern_sv = "single_view_pooled_reference"

    comparison_path = results_dir / csv_comparison
    checkpoint_path = results_dir / ckpt_name

    block = m8_single_view_block_freeze()
    arch = block.architecture
    x_field = str(block.x_field)
    image_normalize = str(block.image_normalize)
    optical_id = str(block.optical_setup_id_reference)
    lr_stage_a = float(
        block.lr_by_role()["primary" if family == "fourier" else "negative_control"]
    )

    catalog_jobs = load_catalog_jobs(
        cfg.workbook_path, cfg.data_root, stl_root=cfg.stl_root
    )
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
            f"M9 e2e {family}: views={view_angles}  epochs={cfg.num_epochs}  "
            f"patience={cfg.early_stop_patience}  batch={cfg.batch_size}  "
            f"lr_stage_a={lr_stage_a:g}  SKIP_TRAIN={skip_train}",
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

        def make_compact() -> nn.Module:
            trunk = make_trunk()
            trunk.load_state_dict(stage_a_state)
            return GeometryAwareFourierFusionLocalizer.for_09_2(
                trunk,
                n_views=len(view_angles),
                view_angles_deg=view_angles,
                fusion_hidden=int(cfg.fusion_hidden),
                fusion_depth=int(cfg.fusion_depth),
            ).to(device)

        def make_large() -> nn.Module:
            trunk = make_trunk()
            trunk.load_state_dict(stage_a_state)
            return GeometryAwareFourierFusionLocalizer.for_09_3(
                trunk,
                n_views=len(view_angles),
                view_angles_deg=view_angles,
                fusion_hidden=int(cfg.fusion_hidden_large),
                fusion_depth=int(cfg.fusion_depth_large),
            ).to(device)

    else:
        backbone = new_frozen_pooled_single_view_expert(
            n_outputs=3, embed_dim=int(arch.head_hidden)
        ).to(device)

        def make_trunk() -> nn.Module:
            return new_frozen_pooled_single_view_expert(
                n_outputs=3, embed_dim=int(arch.head_hidden)
            ).to(device)

        def make_compact() -> nn.Module:
            trunk = make_trunk()
            trunk.load_state_dict(stage_a_state)
            return GeometryAwareFourierFusionLocalizer.for_09_2_pooled(
                trunk,
                n_views=len(view_angles),
                view_angles_deg=view_angles,
                fusion_hidden=int(cfg.fusion_hidden),
                fusion_depth=int(cfg.fusion_depth),
            ).to(device)

        def make_large() -> nn.Module:
            trunk = make_trunk()
            trunk.load_state_dict(stage_a_state)
            return GeometryAwareFourierFusionLocalizer.for_09_3_pooled(
                trunk,
                n_views=len(view_angles),
                view_angles_deg=view_angles,
                fusion_hidden=int(cfg.fusion_hidden_large),
                fusion_depth=int(cfg.fusion_depth_large),
            ).to(device)

    stage_a_state: dict[str, torch.Tensor] | None = None
    fit_a: dict[str, Any] = {
        "history": [],
        "parameter_count": int(
            sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        ),
    }
    fit_compact: dict[str, Any] = {"history": []}
    fit_large: dict[str, Any] = {"history": []}

    if skip_train and checkpoint_path.is_file():
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        stage_a_state = blob.get(ckpt_stage_key)
        if stage_a_state is not None:
            backbone.load_state_dict(stage_a_state)
            if cfg.verbose:
                print(
                    f"Loaded Stage A from {display_path(checkpoint_path)}",
                    flush=True,
                )
        else:
            skip_train = False

    if not skip_train:
        train_ds_sv = ConcatDataset(
            [make_split("train", theta) for theta in view_angles]
        )
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
                f"Stage A e2e {family} train={len(train_ds_sv)}  "
                f"val={len(val_ds_sv)}",
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
            progress_label=f"Stage A e2e {family}",
        )
        stage_a_state = copy.deepcopy(backbone.state_dict())
    elif stage_a_state is None:
        stage_a_state = copy.deepcopy(backbone.state_dict())

    assert stage_a_state is not None

    f_compact = f_large = None
    compact_describe: dict[str, Any] = {}
    large_describe: dict[str, Any] = {}

    if skip_train:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if comparison_path.is_file():
            comparison_df = pd.read_csv(comparison_path)
        elif blob.get("comparison") is not None:
            comparison_df = pd.DataFrame(blob["comparison"])
            comparison_df.to_csv(comparison_path, index=False)
        else:
            raise FileNotFoundError(
                f"M9 e2e checkpoint {display_path(checkpoint_path)} "
                "has no comparison table"
            )
        if "display_label" not in comparison_df.columns:
            comparison_df = comparison_df.copy()
            comparison_df["display_label"] = comparison_df["variant_id"].map(
                lambda v: display_map.get(v, v)
            )
        f_compact = make_compact()
        f_large = make_large()
        if blob.get("f2_state") is not None:
            f_compact.load_state_dict(blob["f2_state"])
        if blob.get("f3_state") is not None:
            f_large.load_state_dict(blob["f3_state"])
        compact_describe = f_compact.describe()
        large_describe = f_large.describe()
        if cfg.verbose:
            print(
                f"SKIP_TRAIN: loaded {display_path(checkpoint_path)} "
                f"({len(comparison_df)} comparison rows)",
                flush=True,
            )
    else:
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
        f_compact = make_compact()
        f_large = make_large()
        if cfg.verbose:
            print(f"09_2 {family}:", f_compact.describe(), flush=True)
            print(f"09_3 {family}:", f_large.describe(), flush=True)
            print(
                f"Stage B e2e {family} train={len(train_ds_mv)}  "
                f"val={len(val_ds_mv)}",
                flush=True,
            )
        fit_compact = train_full_split(
            model=f_compact,
            train_loader=train_loader_mv,
            batch_xy=batch_xy_multiview,
            device=device,
            num_epochs=int(cfg.num_epochs),
            lr=lr_stage_a,
            val_loader=val_loader_mv,
            y_fields=Y_FIELDS,
            early_stop_patience=int(cfg.early_stop_patience),
            progress_label=f"09_2 e2e {family} compact",
        )
        fit_large = train_full_split(
            model=f_large,
            train_loader=train_loader_mv,
            batch_xy=batch_xy_multiview,
            device=device,
            num_epochs=int(cfg.num_epochs),
            lr=lr_stage_a,
            val_loader=val_loader_mv,
            y_fields=Y_FIELDS,
            early_stop_patience=int(cfg.early_stop_patience),
            progress_label=f"09_3 e2e {family} large",
        )
        compact_describe = f_compact.describe()
        large_describe = f_large.describe()

        backbone.load_state_dict(stage_a_state)
        backbone.eval()
        f_compact.eval()
        f_large.eval()
        sv_angle = (
            180.0
            if 180.0 in view_angles
            else float(view_angles[len(view_angles) // 2])
        )

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
            pred_c, _ = _collect_mv_predictions(
                mv_ds,
                x_field=x_field,
                device=device,
                predict_fn=lambda v: f_compact(v),
            )
            pred_l, _ = _collect_mv_predictions(
                mv_ds,
                x_field=x_field,
                device=device,
                predict_fn=lambda v: f_large(v),
            )
            m_xyz = per_axis_rmse(pred_xyz, tgt, Y_FIELDS)
            m_c = per_axis_rmse(pred_c, tgt, Y_FIELDS)
            m_l = per_axis_rmse(pred_l, tgt, Y_FIELDS)
            n_backbone = int(sum(p.numel() for p in backbone.parameters()))
            rows.extend(
                [
                    {
                        "variant_id": id_sv,
                        "display_label": display_sv,
                        "fusion_pattern": pattern_sv,
                        "backbone_kind": backbone_kind,
                        "n_views": 1,
                        "split": split,
                        "encoder_frozen": False,
                        "geometry_tokens": False,
                        "fusion_hidden": None,
                        "fusion_depth": None,
                        "RMSE_total": sv["train_RMSE_total"],
                        "RMSE_X": sv["train_RMSE_X"],
                        "RMSE_Y": sv["train_RMSE_Y"],
                        "RMSE_Z": sv["train_RMSE_Z"],
                        "learned_parameter_count": n_backbone,
                        "fusion_parameter_count": None,
                    },
                    {
                        "variant_id": id_xyz,
                        "display_label": display_xyz,
                        "fusion_pattern": "shared_xyz_mean",
                        "backbone_kind": backbone_kind,
                        "n_views": len(view_angles),
                        "split": split,
                        "encoder_frozen": True,
                        "geometry_tokens": False,
                        "fusion_hidden": None,
                        "fusion_depth": None,
                        "RMSE_total": m_xyz["train_RMSE_total"],
                        "RMSE_X": m_xyz["train_RMSE_X"],
                        "RMSE_Y": m_xyz["train_RMSE_Y"],
                        "RMSE_Z": m_xyz["train_RMSE_Z"],
                        "learned_parameter_count": n_backbone,
                        "fusion_parameter_count": None,
                    },
                    {
                        "variant_id": id_compact,
                        "display_label": display_compact,
                        "fusion_pattern": pattern_compact,
                        "backbone_kind": backbone_kind,
                        "n_views": len(view_angles),
                        "split": split,
                        "encoder_frozen": False,
                        "geometry_tokens": True,
                        "fusion_hidden": int(f_compact.fusion_hidden),
                        "fusion_depth": int(f_compact.fusion_depth),
                        "RMSE_total": m_c["train_RMSE_total"],
                        "RMSE_X": m_c["train_RMSE_X"],
                        "RMSE_Y": m_c["train_RMSE_Y"],
                        "RMSE_Z": m_c["train_RMSE_Z"],
                        "learned_parameter_count": int(
                            f_compact.learned_parameter_count()
                        ),
                        "fusion_parameter_count": int(
                            f_compact.fusion_parameter_count()
                        ),
                    },
                    {
                        "variant_id": id_large,
                        "display_label": display_large,
                        "fusion_pattern": pattern_large,
                        "backbone_kind": backbone_kind,
                        "n_views": len(view_angles),
                        "split": split,
                        "encoder_frozen": False,
                        "geometry_tokens": True,
                        "fusion_hidden": int(f_large.fusion_hidden),
                        "fusion_depth": int(f_large.fusion_depth),
                        "RMSE_total": m_l["train_RMSE_total"],
                        "RMSE_X": m_l["train_RMSE_X"],
                        "RMSE_Y": m_l["train_RMSE_Y"],
                        "RMSE_Z": m_l["train_RMSE_Z"],
                        "learned_parameter_count": int(
                            f_large.learned_parameter_count()
                        ),
                        "fusion_parameter_count": int(
                            f_large.fusion_parameter_count()
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

        payload = {
            "experiment_id": experiment_id,
            "family": family,
            "n_views": len(view_angles),
            "view_angles_deg": list(view_angles),
            "angle_stride_deg": float(cfg.angle_stride_deg),
            "lr_stage_a": lr_stage_a,
            "num_epochs": int(cfg.num_epochs),
            "early_stop_patience": int(cfg.early_stop_patience),
            "batch_size": int(cfg.batch_size),
            "f2": compact_describe,
            "f3": large_describe,
            "stage_a_epochs": len(fit_a["history"]),
            "f2_epochs": len(fit_compact["history"]),
            "f3_epochs": len(fit_large["history"]),
            "comparison": comparison_df.to_dict(orient="records"),
        }
        (results_dir / json_name).write_text(
            json.dumps(payload, indent=2, default=str)
        )
        save_study_checkpoint(
            checkpoint_path,
            {
                ckpt_stage_key: stage_a_state,
                "f2_state": clone_state_dict(f_compact),
                "f3_state": clone_state_dict(f_large),
                "comparison": comparison_df.to_dict(orient="records"),
                "experiment_id": experiment_id,
                "family": family,
                "view_angles_deg": list(view_angles),
                "lr_stage_a": lr_stage_a,
            },
        )
        if cfg.verbose:
            print(f"Wrote {display_path(comparison_path)}", flush=True)
            print(f"Wrote {display_path(checkpoint_path)}", flush=True)

    return M9E2EFamilyResult(
        family=family,
        comparison_df=comparison_df,
        view_angles=view_angles,
        lr_stage_a=lr_stage_a,
        results_dir=results_dir,
        comparison_path=comparison_path,
        checkpoint_path=checkpoint_path,
        skipped_train=bool(skip_train),
        num_epochs=int(cfg.num_epochs),
        early_stop_patience=int(cfg.early_stop_patience),
        batch_size=int(cfg.batch_size),
        compact_describe=compact_describe,
        large_describe=large_describe,
        extra={
            "stage_a_epochs": len(fit_a.get("history", [])),
            "f2_epochs": len(fit_compact.get("history", [])),
            "f3_epochs": len(fit_large.get("history", [])),
        },
    )
