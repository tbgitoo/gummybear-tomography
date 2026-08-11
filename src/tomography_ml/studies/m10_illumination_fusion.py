"""M10 illumination-only fusion (10_1A frozen + 10_1B e2e).

Stage A trains Fourier and optional pooled single-view trunks on concatenated
single-light datasets; Stage B trains C/D fusion heads (Fourier + optional
pooled) with optional LR sweeps; evaluation covers A/B/C/D baselines on
train/validation/test illumination stacks.

Matches ``notebooks/10_1A_illumination_only_frozen_fusion.ipynb`` and
``notebooks/10_1B_illumination_only_e2e_fusion.ipynb``.
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

from gummybear.paths import display_path, repo_relative_path
from tomography_ml.gummybear_data_catalog import (
    IlluminationOnlyDataset,
    build_catalog_rows,
    build_illumination_joint_groups,
    count_groups_by_split,
    groups_for_split,
    load_catalog_jobs,
)
from tomography_ml.localization.architecture_capability import (
    evaluate_split_rmse,
    train_full_split,
)
from tomography_ml.localization.builders import m8_single_view_block_freeze
from tomography_ml.localization.localize_multiview import (
    FUSION_PATTERN_10_1_C,
    FUSION_PATTERN_10_1_C_FROZEN,
    FUSION_PATTERN_10_1_C_FROZEN_POOLED,
    FUSION_PATTERN_10_1_C_POOLED,
    FUSION_PATTERN_10_1_D,
    FUSION_PATTERN_10_1_D_FROZEN,
    FUSION_PATTERN_10_1_D_FROZEN_POOLED,
    FUSION_PATTERN_10_1_D_POOLED,
    GEOMETRY_MODE_CONCAT,
    GEOMETRY_MODE_FILM,
    M10_LIGHT_ANGLES_DEG,
    CompactLatentFusionLocalizer,
    GeometryAwareFourierFusionLocalizer,
    new_frozen_pooled_single_view_expert,
    new_frozen_single_view_expert,
    shared_xyz_mean,
)
from tomography_ml.studies.m9_frozen_fusion import DEFAULT_LR_STAGE_B_GRID
from tomography_ml.studies.study_checkpoints import (
    M10_E2E_ILLUMINATION_FUSION,
    M10_FROZEN_ILLUMINATION_FUSION,
    clone_state_dict,
    load_study_checkpoint,
    save_study_checkpoint,
)
from tomography_ml.training.training_helpers import (
    eval_stack,
    make_batch_xy_single,
    make_sv_illumination_dataset,
    persist_stage_b_lr_artifacts,
    resolve_run_lr_study,
    run_stage_b_lr_study,
)
from tomography_ml_validation.run_history import (
    aggregate_run_history,
    append_run_history,
    build_history_row,
    effective_n_repeat,
    next_run_id,
    utc_now_iso,
)

Mode = Literal["frozen", "e2e"]
MakeModelFn = Callable[[], nn.Module]

Y_FIELDS: tuple[str, ...] = ("particle_x", "particle_y", "particle_z")
DEFAULT_LR_STAGE_B: float = 3e-3

_BASE_A = "m10_1a_single_illumination"
_BASE_B = "m10_1b_mean_xyz_illuminations"


@dataclass(frozen=True)
class _M10ArtifactNames:
    """Notebook-compatible CSV/JSON names plus milestone checkpoint filename."""

    file_prefix: str
    notebook_id: str
    experiment_id: str
    checkpoint_name: str
    json_name: str
    csv_comparison: str
    csv_lr_study: str
    csv_lr_study_fourier: str
    csv_lr_study_pooled: str
    json_recommended_lrs: str
    csv_run_history: str
    csv_session_summary: str


def _artifact_names(mode: Mode) -> _M10ArtifactNames:
    prefix = "m10_1a" if mode == "frozen" else "m10_1b"
    exp = (
        "m10_1a_illumination_only_frozen_fusion"
        if mode == "frozen"
        else "m10_1b_illumination_only_e2e_fusion"
    )
    json_stem = (
        "m10_1a_illumination_only_fusion"
        if mode == "frozen"
        else "m10_1b_illumination_only_fusion"
    )
    ckpt = (
        M10_FROZEN_ILLUMINATION_FUSION
        if mode == "frozen"
        else M10_E2E_ILLUMINATION_FUSION
    )
    return _M10ArtifactNames(
        file_prefix=prefix,
        notebook_id="10_1A" if mode == "frozen" else "10_1B",
        experiment_id=exp,
        checkpoint_name=ckpt,
        json_name=f"{json_stem}.json",
        csv_comparison=f"{prefix}_comparison.csv",
        csv_lr_study=f"{prefix}_lr_study.csv",
        csv_lr_study_fourier=f"{prefix}_lr_study_fourier.csv",
        csv_lr_study_pooled=f"{prefix}_lr_study_pooled.csv",
        json_recommended_lrs=f"{prefix}_recommended_lrs.json",
        csv_run_history=f"{prefix}_run_history.csv",
        csv_session_summary=f"{prefix}_session_summary.csv",
    )


@dataclass
class M10IlluminationConfig:
    """Hyperparameters and paths for one M10 illumination-fusion run."""

    mode: Mode
    workbook_path: Path
    data_root: Path
    results_dir: Path
    stl_root: Path
    device: torch.device | str
    num_epochs: int = 200
    early_stop_patience: int = 40
    batch_size: int = 16
    fixed_camera_deg: float = 180.0
    single_light_ref_deg: float = 0.0
    include_pooled_control: bool = True
    n_repeat_training: int = 1
    run_lr_study: bool | str = False
    # LR sweeps remain illustrative; reported Stage-B models use lr_stage_b_*.
    select_best_val_lr: bool = False
    lr_stage_b_grid: Sequence[float] = DEFAULT_LR_STAGE_B_GRID
    lr_stage_b_c_fourier: float = DEFAULT_LR_STAGE_B
    lr_stage_b_d_fourier: float = DEFAULT_LR_STAGE_B
    lr_stage_b_c_pooled: float = DEFAULT_LR_STAGE_B
    lr_stage_b_d_pooled: float = DEFAULT_LR_STAGE_B
    lr_stage_a_pooled: float | None = None
    use_angle_film: bool = True
    light_angles_deg: Sequence[float] | None = None
    min_joint_groups: int = 4
    load_existing: bool = True
    retrain: bool = False
    verbose: bool = True


@dataclass
class M10IlluminationResult:
    """Artifacts from :func:`run_m10_illumination_fusion`."""

    mode: Mode
    comparison_df: pd.DataFrame
    session_summary_df: pd.DataFrame
    lr_study_df: pd.DataFrame
    run_history_df: pd.DataFrame
    results_dir: Path
    comparison_path: Path
    checkpoint_path: Path
    session_run_ids: list[int]
    skipped_train: bool
    selected_lrs: dict[str, Any]
    session_summary_path: Path = field(default_factory=Path)
    history_path: Path = field(default_factory=Path)
    lr_study_path: Path = field(default_factory=Path)
    json_path: Path = field(default_factory=Path)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _FusionHeadSpec:
    variant_id: str
    fusion_pattern: str
    use_angles: bool
    lr_key: str


def _fusion_head_specs(mode: Mode) -> tuple[_FusionHeadSpec, _FusionHeadSpec]:
    if mode == "frozen":
        return (
            _FusionHeadSpec(
                "m10_1a_c_frozen_illumination_fusion",
                FUSION_PATTERN_10_1_C_FROZEN,
                False,
                "LR_STAGE_B_C_FOURIER",
            ),
            _FusionHeadSpec(
                "m10_1a_d_frozen_illumination_angle_fusion",
                FUSION_PATTERN_10_1_D_FROZEN,
                True,
                "LR_STAGE_B_D_FOURIER",
            ),
        )
    return (
        _FusionHeadSpec(
            "m10_1c_e2e_illumination_fusion",
            FUSION_PATTERN_10_1_C,
            False,
            "LR_STAGE_B_C_FOURIER",
        ),
        _FusionHeadSpec(
            "m10_1d_e2e_illumination_angle_fusion",
            FUSION_PATTERN_10_1_D,
            True,
            "LR_STAGE_B_D_FOURIER",
        ),
    )


def _fusion_head_specs_pooled(mode: Mode) -> tuple[_FusionHeadSpec, _FusionHeadSpec]:
    if mode == "frozen":
        return (
            _FusionHeadSpec(
                "m10_1a_c_frozen_illumination_fusion_pooled",
                FUSION_PATTERN_10_1_C_FROZEN_POOLED,
                False,
                "LR_STAGE_B_C_POOLED",
            ),
            _FusionHeadSpec(
                "m10_1a_d_frozen_illumination_angle_fusion_pooled",
                FUSION_PATTERN_10_1_D_FROZEN_POOLED,
                True,
                "LR_STAGE_B_D_POOLED",
            ),
        )
    return (
        _FusionHeadSpec(
            "m10_1c_e2e_illumination_fusion_pooled",
            FUSION_PATTERN_10_1_C_POOLED,
            False,
            "LR_STAGE_B_C_POOLED",
        ),
        _FusionHeadSpec(
            "m10_1d_e2e_illumination_angle_fusion_pooled",
            FUSION_PATTERN_10_1_D_POOLED,
            True,
            "LR_STAGE_B_D_POOLED",
        ),
    )


def _resolve_lr_plan(
    cfg: M10IlluminationConfig,
    names: _M10ArtifactNames,
) -> dict[str, Any]:
    lr_defaults = {
        "LR_STAGE_B_C_FOURIER": float(cfg.lr_stage_b_c_fourier),
        "LR_STAGE_B_D_FOURIER": float(cfg.lr_stage_b_d_fourier),
        "LR_STAGE_B_C_POOLED": float(cfg.lr_stage_b_c_pooled),
        "LR_STAGE_B_D_POOLED": float(cfg.lr_stage_b_d_pooled),
    }
    if cfg.mode == "frozen":
        return resolve_run_lr_study(
            cfg.run_lr_study,
            recommended_path=cfg.results_dir / names.json_recommended_lrs,
            use_angle_film=bool(cfg.use_angle_film),
            include_pooled_control=bool(cfg.include_pooled_control),
            lr_defaults=lr_defaults,
        )
    run_flag = bool(cfg.run_lr_study) if not isinstance(cfg.run_lr_study, str) else False
    return {
        "mode": "always" if run_flag else "never",
        "effective_run_lr_study": run_flag,
        "study_by_key": {
            k: run_flag
            for k in (
                "LR_STAGE_B_C_FOURIER",
                "LR_STAGE_B_D_FOURIER",
                *(
                    ("LR_STAGE_B_C_POOLED", "LR_STAGE_B_D_POOLED")
                    if cfg.include_pooled_control
                    else ()
                ),
            )
        },
        "lrs": dict(lr_defaults),
        "loaded_recommended": None,
        "known_keys": [],
        "unknown_keys": [],
        "use_angle_film": bool(cfg.use_angle_film),
    }


def _make_model_factories(
    *,
    mode: Mode,
    device: torch.device | str,
    arch_hidden: int,
    n_lights: int,
    light_angles_deg: Sequence[float],
    geometry_mode_d: str,
    stage_a_state: dict[str, torch.Tensor],
    stage_a_state_p: dict[str, torch.Tensor] | None,
) -> dict[str, Callable[[], nn.Module]]:
    lights = tuple(float(a) for a in light_angles_deg)

    def make_trunk_f(state: dict[str, torch.Tensor] = stage_a_state) -> nn.Module:
        trunk = new_frozen_single_view_expert(n_outputs=3, hidden=int(arch_hidden)).to(
            device
        )
        trunk.load_state_dict(state)
        return trunk

    def make_trunk_p(
        state: dict[str, torch.Tensor] | None = stage_a_state_p,
    ) -> nn.Module:
        assert state is not None
        trunk = new_frozen_pooled_single_view_expert(
            n_outputs=3, embed_dim=int(arch_hidden)
        ).to(device)
        trunk.load_state_dict(state)
        return trunk

    factories: dict[str, Callable[[], nn.Module]] = {}
    if mode == "frozen":
        factories["c_fourier"] = lambda: CompactLatentFusionLocalizer.for_10_1_c_frozen(
            make_trunk_f(), n_views=n_lights
        ).to(device)
        factories["d_fourier"] = lambda: GeometryAwareFourierFusionLocalizer.for_10_1_d_frozen(
            make_trunk_f(),
            n_views=n_lights,
            light_angles_deg=lights,
            geometry_mode=geometry_mode_d,
        ).to(device)
        if stage_a_state_p is not None:
            factories["c_pooled"] = (
                lambda: CompactLatentFusionLocalizer.for_10_1_c_frozen_pooled(
                    make_trunk_p(), n_views=n_lights
                ).to(device)
            )
            factories["d_pooled"] = (
                lambda: GeometryAwareFourierFusionLocalizer.for_10_1_d_frozen_pooled(
                    make_trunk_p(),
                    n_views=n_lights,
                    light_angles_deg=lights,
                    geometry_mode=geometry_mode_d,
                ).to(device)
            )
    else:
        factories["c_fourier"] = lambda: CompactLatentFusionLocalizer.for_10_1_c(
            make_trunk_f(), n_views=n_lights
        ).to(device)
        factories["d_fourier"] = lambda: GeometryAwareFourierFusionLocalizer.for_10_1_d(
            make_trunk_f(),
            n_views=n_lights,
            light_angles_deg=lights,
        ).to(device)
        if stage_a_state_p is not None:
            factories["c_pooled"] = lambda: CompactLatentFusionLocalizer.for_10_1_c_pooled(
                make_trunk_p(), n_views=n_lights
            ).to(device)
            factories["d_pooled"] = (
                lambda: GeometryAwareFourierFusionLocalizer.for_10_1_d_pooled(
                    make_trunk_p(),
                    n_views=n_lights,
                    light_angles_deg=lights,
                ).to(device)
            )
    return factories


def _load_existing_result(
    cfg: M10IlluminationConfig,
    names: _M10ArtifactNames,
    checkpoint_path: Path,
) -> M10IlluminationResult:
    blob = load_study_checkpoint(checkpoint_path)
    comparison_path = cfg.results_dir / names.csv_comparison
    session_summary_path = cfg.results_dir / names.csv_session_summary
    lr_study_path = cfg.results_dir / names.csv_lr_study
    history_path = cfg.results_dir / names.csv_run_history

    if comparison_path.is_file():
        comparison_df = pd.read_csv(comparison_path)
    elif blob.get("comparison_last_run") is not None:
        comparison_df = pd.DataFrame(blob["comparison_last_run"])
        comparison_df.to_csv(comparison_path, index=False)
    elif blob.get("comparison") is not None:
        comparison_df = pd.DataFrame(blob["comparison"])
        comparison_df.to_csv(comparison_path, index=False)
    else:
        raise FileNotFoundError(
            f"M10 checkpoint {display_path(checkpoint_path)} has no comparison table"
        )

    if session_summary_path.is_file():
        session_summary_df = pd.read_csv(session_summary_path)
    elif blob.get("session_summary") is not None:
        session_summary_df = pd.DataFrame(blob["session_summary"])
        session_summary_df.to_csv(session_summary_path, index=False)
    else:
        session_summary_df = pd.DataFrame()

    if lr_study_path.is_file():
        lr_study_df = pd.read_csv(lr_study_path)
    elif blob.get("lr_study") is not None:
        lr_study_df = pd.DataFrame(blob["lr_study"])
        lr_study_df.to_csv(lr_study_path, index=False)
    else:
        lr_study_df = pd.DataFrame()

    if history_path.is_file():
        run_history_df = pd.read_csv(history_path)
    else:
        run_history_df = pd.DataFrame()

    selected = blob.get("selected_lrs")
    if not isinstance(selected, dict):
        selected = {
            "LR_STAGE_B_C_FOURIER": blob.get("selected_lr_c"),
            "LR_STAGE_B_D_FOURIER": blob.get("selected_lr_d"),
            "LR_STAGE_B_C_POOLED": blob.get("selected_lr_c_p"),
            "LR_STAGE_B_D_POOLED": blob.get("selected_lr_d_p"),
        }

    session_run_ids = list(blob.get("session_run_ids") or [])
    if cfg.verbose:
        print(
            f"SKIP_TRAIN: loaded {display_path(checkpoint_path)} "
            f"({len(comparison_df)} comparison rows)",
            flush=True,
        )

    return M10IlluminationResult(
        mode=cfg.mode,
        comparison_df=comparison_df,
        session_summary_df=session_summary_df,
        lr_study_df=lr_study_df,
        run_history_df=run_history_df,
        results_dir=Path(cfg.results_dir),
        comparison_path=comparison_path,
        checkpoint_path=checkpoint_path,
        session_run_ids=session_run_ids,
        skipped_train=True,
        selected_lrs=dict(selected),
        session_summary_path=session_summary_path,
        history_path=history_path,
        lr_study_path=lr_study_path,
        json_path=cfg.results_dir / names.json_name,
        extra={
            "experiment_id": names.experiment_id,
            "loaded_from_checkpoint": True,
        },
    )


def run_m10_illumination_fusion(cfg: M10IlluminationConfig) -> M10IlluminationResult:
    """Run (or load) M10 illumination-only fusion for frozen or e2e mode."""
    names = _artifact_names(cfg.mode)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device
    checkpoint_path = results_dir / names.checkpoint_name
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

    block = m8_single_view_block_freeze()
    arch = block.architecture
    x_field = str(block.x_field)
    image_normalize = str(block.image_normalize)
    roles = block.lr_by_role()
    lr_fourier = float(roles["primary"])
    lr_pooled_a = float(
        cfg.lr_stage_a_pooled
        if cfg.lr_stage_a_pooled is not None
        else roles.get("negative_control", roles["primary"])
    )

    lr_plan = _resolve_lr_plan(cfg, names)
    lr_stage_b_c_fourier = float(lr_plan["lrs"]["LR_STAGE_B_C_FOURIER"])
    lr_stage_b_d_fourier = float(lr_plan["lrs"]["LR_STAGE_B_D_FOURIER"])
    lr_stage_b_c_pooled = float(lr_plan["lrs"]["LR_STAGE_B_C_POOLED"])
    lr_stage_b_d_pooled = float(lr_plan["lrs"]["LR_STAGE_B_D_POOLED"])
    effective_run_lr_study = bool(lr_plan["effective_run_lr_study"])
    lr_study_by_key: dict[str, bool] = dict(lr_plan["study_by_key"])
    geometry_mode_d = GEOMETRY_MODE_FILM if cfg.use_angle_film else GEOMETRY_MODE_CONCAT

    skip_train = bool(
        cfg.load_existing and checkpoint_path.is_file() and not cfg.retrain
    )

    if cfg.verbose:
        print(
            f"M10 {cfg.mode}: lights={light_angles}  cam={cfg.fixed_camera_deg}  "
            f"epochs={cfg.num_epochs}  patience={cfg.early_stop_patience}  "
            f"batch={cfg.batch_size}  film={cfg.use_angle_film}  "
            f"RUN_LR_STUDY={cfg.run_lr_study!r}→{effective_run_lr_study}  "
            f"SKIP_TRAIN={skip_train}",
            flush=True,
        )
        print(f"results={repo_relative_path(results_dir)}", flush=True)

    if skip_train:
        return _load_existing_result(cfg, names, checkpoint_path)

    catalog_jobs = load_catalog_jobs(cfg.workbook_path, cfg.data_root, stl_root=cfg.stl_root)
    catalog_rows = build_catalog_rows(catalog_jobs)
    joint_groups = build_illumination_joint_groups(
        catalog_rows,
        light_angles_deg=light_angles,
        min_groups=int(cfg.min_joint_groups),
    )
    if cfg.verbose:
        print(
            f"joint groups={len(joint_groups)}  splits={count_groups_by_split(joint_groups)}",
            flush=True,
        )

    n_lights = len(light_angles)
    train_ds = IlluminationOnlyDataset(
        groups_for_split(joint_groups, "train"),
        x_field=x_field,
        y_fields=Y_FIELDS,
        fixed_camera_deg=cfg.fixed_camera_deg,
        light_angles_deg=light_angles,
        image_normalize=image_normalize,
    )
    val_ds = IlluminationOnlyDataset(
        groups_for_split(joint_groups, "validation"),
        x_field=x_field,
        y_fields=Y_FIELDS,
        fixed_camera_deg=cfg.fixed_camera_deg,
        light_angles_deg=light_angles,
        image_normalize=image_normalize,
    )
    test_ds = IlluminationOnlyDataset(
        groups_for_split(joint_groups, "test"),
        x_field=x_field,
        y_fields=Y_FIELDS,
        fixed_camera_deg=cfg.fixed_camera_deg,
        light_angles_deg=light_angles,
        image_normalize=image_normalize,
    )

    def make_sv_dataset(split: str, light_angle_deg: float) -> Dataset:
        return make_sv_illumination_dataset(
            catalog_rows,
            split=split,
            light_angle_deg=light_angle_deg,
            x_field=x_field,
            y_fields=Y_FIELDS,
            fixed_camera_deg=cfg.fixed_camera_deg,
            image_normalize=image_normalize,
        )

    batch_xy_single = make_batch_xy_single(
        x_field=x_field, y_fields=Y_FIELDS, device=device
    )
    train_ds_sv = ConcatDataset(
        [make_sv_dataset("train", L) for L in light_angles]
    )
    val_ds_sv = ConcatDataset(
        [make_sv_dataset("validation", L) for L in light_angles]
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

    def stage_b(
        *,
        tag: str,
        make_model: MakeModelFn,
        use_angles: bool,
        lr_fixed: float,
        run_study: bool,
    ) -> dict[str, Any]:
        return run_stage_b_lr_study(
            tag=tag,
            make_model=make_model,
            train_ds=train_ds,
            val_ds=val_ds,
            y_fields=Y_FIELDS,
            device=device,
            use_angles=use_angles,
            lr_fixed=lr_fixed,
            run_study=run_study,
            select_best_val_lr=bool(cfg.select_best_val_lr),
            lr_grid=tuple(float(x) for x in cfg.lr_stage_b_grid),
            num_epochs=int(cfg.num_epochs),
            early_stop_patience=int(cfg.early_stop_patience),
            batch_size=int(cfg.batch_size),
        )

    n_rep = effective_n_repeat(
        int(cfg.n_repeat_training), run_lr_study=effective_run_lr_study
    )
    if effective_run_lr_study and int(cfg.n_repeat_training) > 1 and cfg.verbose:
        print(
            "NOTE: LR study active → forcing a single training repeat "
            f"(requested n_repeat_training={cfg.n_repeat_training}).",
            flush=True,
        )

    head_c, head_d = _fusion_head_specs(cfg.mode)
    head_c_p, head_d_p = _fusion_head_specs_pooled(cfg.mode)
    history_path = results_dir / names.csv_run_history

    session_run_ids: list[int] = []
    session_history_rows: list[dict[str, Any]] = []
    last_comparison_rows: list[dict[str, Any]] | None = None
    last_lr_study_fourier_df: pd.DataFrame | None = None
    last_lr_study_pooled_df = pd.DataFrame()
    last_payload: dict[str, Any] = {}

    lr_c_f = lr_stage_b_c_fourier
    lr_d_f = lr_stage_b_d_fourier
    lr_c_p_fixed = lr_stage_b_c_pooled
    lr_d_p_fixed = lr_stage_b_d_pooled

    for rep in range(n_rep):
        run_id = max(session_run_ids) + 1 if session_run_ids else next_run_id(history_path)
        session_run_ids.append(run_id)
        ts = utc_now_iso()

        if cfg.mode == "frozen":
            study_c_f = bool(lr_study_by_key.get("LR_STAGE_B_C_FOURIER") and rep == 0)
            study_d_f = bool(lr_study_by_key.get("LR_STAGE_B_D_FOURIER") and rep == 0)
            study_c_p = bool(lr_study_by_key.get("LR_STAGE_B_C_POOLED") and rep == 0)
            study_d_p = bool(lr_study_by_key.get("LR_STAGE_B_D_POOLED") and rep == 0)
            do_lr_study = bool(study_c_f or study_d_f or study_c_p or study_d_p)
        else:
            do_lr_study = bool(effective_run_lr_study and rep == 0)
            study_c_f = study_d_f = study_c_p = study_d_p = do_lr_study

        if cfg.verbose:
            print(
                f"\n######## Train+eval repeat {rep + 1}/{n_rep}  "
                f"run_id={run_id}  lr_study={do_lr_study} ########",
                flush=True,
            )

        backbone = new_frozen_single_view_expert(
            n_outputs=3, hidden=int(arch.head_hidden)
        ).to(device)
        if cfg.verbose:
            print(
                f"=== Stage A Fourier  n_train={len(train_ds_sv)}  "
                f"n_val={len(val_ds_sv)}  epochs≤{cfg.num_epochs} ===",
                flush=True,
            )
        fit_a = train_full_split(
            model=backbone,
            train_loader=train_loader_sv,
            batch_xy=batch_xy_single,
            device=device,
            num_epochs=int(cfg.num_epochs),
            lr=lr_fourier,
            val_loader=val_loader_sv,
            y_fields=Y_FIELDS,
            early_stop_patience=int(cfg.early_stop_patience),
            progress_label="Stage A Fourier",
        )
        stage_a_state = copy.deepcopy(backbone.state_dict())
        epochs_a_f = len(fit_a["history"])

        factories = _make_model_factories(
            mode=cfg.mode,
            device=device,
            arch_hidden=int(arch.head_hidden),
            n_lights=n_lights,
            light_angles_deg=light_angles,
            geometry_mode_d=geometry_mode_d,
            stage_a_state=stage_a_state,
            stage_a_state_p=None,
        )
        result_c = stage_b(
            tag="C_Fourier",
            make_model=factories["c_fourier"],
            use_angles=False,
            lr_fixed=lr_c_f,
            run_study=study_c_f,
        )
        result_d = stage_b(
            tag="D_Fourier",
            make_model=factories["d_fourier"],
            use_angles=True,
            lr_fixed=lr_d_f,
            run_study=study_d_f,
        )
        model_c = result_c["model"]
        model_d = result_d["model"]
        fit_c = result_c["fit"]
        fit_d = result_d["fit"]
        selected_lr_c = float(result_c["selected_lr"])
        selected_lr_d = float(result_d["selected_lr"])
        lr_c_f = selected_lr_c
        lr_d_f = selected_lr_d
        last_lr_study_fourier_df = pd.concat(
            [result_c["study_df"], result_d["study_df"]], ignore_index=True
        )

        model_c_p = model_d_p = None
        backbone_p = None
        stage_a_state_p = None
        fit_c_p = fit_d_p = None
        selected_lr_c_p = selected_lr_d_p = None
        epochs_a_p = float("nan")
        last_lr_study_pooled_df = pd.DataFrame()

        if cfg.include_pooled_control:
            backbone_p = new_frozen_pooled_single_view_expert(
                n_outputs=3, embed_dim=int(arch.head_hidden)
            ).to(device)
            if cfg.verbose:
                print(
                    f"=== Stage A pooled  lr={lr_pooled_a:g}  "
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
                lr=lr_pooled_a,
                val_loader=val_loader_sv,
                y_fields=Y_FIELDS,
                early_stop_patience=int(cfg.early_stop_patience),
                progress_label="Stage A pooled",
            )
            stage_a_state_p = copy.deepcopy(backbone_p.state_dict())
            epochs_a_p = len(fit_a_p["history"])
            factories_p = _make_model_factories(
                mode=cfg.mode,
                device=device,
                arch_hidden=int(arch.head_hidden),
                n_lights=n_lights,
                light_angles_deg=light_angles,
                geometry_mode_d=geometry_mode_d,
                stage_a_state=stage_a_state,
                stage_a_state_p=stage_a_state_p,
            )
            result_c_p = stage_b(
                tag="C_pooled",
                make_model=factories_p["c_pooled"],
                use_angles=False,
                lr_fixed=lr_c_p_fixed,
                run_study=study_c_p,
            )
            result_d_p = stage_b(
                tag="D_pooled",
                make_model=factories_p["d_pooled"],
                use_angles=True,
                lr_fixed=lr_d_p_fixed,
                run_study=study_d_p,
            )
            model_c_p = result_c_p["model"]
            model_d_p = result_d_p["model"]
            fit_c_p = result_c_p["fit"]
            fit_d_p = result_d_p["fit"]
            selected_lr_c_p = float(result_c_p["selected_lr"])
            selected_lr_d_p = float(result_d_p["selected_lr"])
            lr_c_p_fixed = selected_lr_c_p
            lr_d_p_fixed = selected_lr_d_p
            last_lr_study_pooled_df = pd.concat(
                [result_c_p["study_df"], result_d_p["study_df"]], ignore_index=True
            )

        if cfg.verbose:
            print("=== Evaluation on train/val/test ===", flush=True)
        backbone.load_state_dict(stage_a_state)
        backbone.eval()

        def eval_single_light(split: str, bb: nn.Module = backbone) -> dict[str, float]:
            ds = make_sv_dataset(split, cfg.single_light_ref_deg)
            loader = DataLoader(
                ds, batch_size=min(int(cfg.batch_size), len(ds)), shuffle=False
            )
            return evaluate_split_rmse(
                model=bb,
                loader=loader,
                batch_xy=batch_xy_single,
                y_fields=Y_FIELDS,
                prefix="train",
            )

        lr_map = {
            head_c.variant_id: selected_lr_c,
            head_d.variant_id: selected_lr_d,
            head_c_p.variant_id: selected_lr_c_p,
            head_d_p.variant_id: selected_lr_d_p,
        }
        metrics_by_variant: dict[str, dict[str, Any]] = {}
        comparison_rows: list[dict[str, Any]] = []

        for split, ds in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
            m_a = eval_single_light(split)
            m_b = eval_stack(
                ds,
                lambda v, _l: shared_xyz_mean(backbone, v),
                y_fields=Y_FIELDS,
                device=device,
                batch_size=int(cfg.batch_size),
            )
            m_c = eval_stack(
                ds,
                lambda v, _l: model_c(v),
                y_fields=Y_FIELDS,
                device=device,
                batch_size=int(cfg.batch_size),
            )
            m_d = eval_stack(
                ds,
                lambda v, l: model_d(v, angles_deg=l),
                y_fields=Y_FIELDS,
                device=device,
                batch_size=int(cfg.batch_size),
            )
            variants: list[tuple[str, str, dict[str, float], int, bool, str, Any]] = [
                (_BASE_A, "single_illumination_reference", m_a, 1, False, "fourier", None),
                (_BASE_B, "shared_xyz_mean", m_b, n_lights, False, "fourier", None),
                (
                    head_c.variant_id,
                    head_c.fusion_pattern,
                    m_c,
                    n_lights,
                    False,
                    "fourier",
                    model_c.describe(),
                ),
                (
                    head_d.variant_id,
                    head_d.fusion_pattern,
                    m_d,
                    n_lights,
                    True,
                    "fourier",
                    model_d.describe(),
                ),
            ]
            if cfg.include_pooled_control and model_c_p is not None and backbone_p is not None:
                backbone_p.load_state_dict(stage_a_state_p)
                backbone_p.eval()
                m_a_p = eval_single_light(split, bb=backbone_p)
                m_b_p = eval_stack(
                    ds,
                    lambda v, _l: shared_xyz_mean(backbone_p, v),
                    y_fields=Y_FIELDS,
                    device=device,
                    batch_size=int(cfg.batch_size),
                )
                m_c_p = eval_stack(
                    ds,
                    lambda v, _l: model_c_p(v),
                    y_fields=Y_FIELDS,
                    device=device,
                    batch_size=int(cfg.batch_size),
                )
                m_d_p = eval_stack(
                    ds,
                    lambda v, l: model_d_p(v, angles_deg=l),
                    y_fields=Y_FIELDS,
                    device=device,
                    batch_size=int(cfg.batch_size),
                )
                variants.extend(
                    [
                        (
                            f"{_BASE_A}_pooled",
                            "single_illumination_reference",
                            m_a_p,
                            1,
                            False,
                            "pooled",
                            None,
                        ),
                        (
                            f"{_BASE_B}_pooled",
                            "shared_xyz_mean",
                            m_b_p,
                            n_lights,
                            False,
                            "pooled",
                            None,
                        ),
                        (
                            head_c_p.variant_id,
                            head_c_p.fusion_pattern,
                            m_c_p,
                            n_lights,
                            False,
                            "pooled",
                            model_c_p.describe(),
                        ),
                        (
                            head_d_p.variant_id,
                            head_d_p.fusion_pattern,
                            m_d_p,
                            n_lights,
                            True,
                            "pooled",
                            model_d_p.describe(),
                        ),
                    ]
                )

            for variant, pattern, m, n_views, geom, trunk, desc in variants:
                bucket = metrics_by_variant.setdefault(
                    variant,
                    {
                        "fusion_pattern": pattern,
                        "backbone_kind": trunk,
                        "n_views": n_views,
                        "light_geometry": geom,
                        "describe": desc,
                        "by_split": {},
                    },
                )
                bucket["by_split"][split] = m
                comparison_rows.append(
                    {
                        "run_id": run_id,
                        "repeat_index": rep,
                        "variant_id": variant,
                        "fusion_pattern": pattern,
                        "backbone_kind": trunk,
                        "split": split,
                        "fixed_camera_deg": cfg.fixed_camera_deg,
                        "n_views": n_views,
                        "n_lights": n_views,
                        "light_geometry": geom,
                        "lr_stage_b": lr_map.get(variant),
                        "RMSE_total": m["train_RMSE_total"],
                        "RMSE_X": m["train_RMSE_X"],
                        "RMSE_Y": m["train_RMSE_Y"],
                        "RMSE_Z": m["train_RMSE_Z"],
                        "learned_parameter_count": (
                            None
                            if desc is None
                            else desc.get("learned_parameter_count")
                        ),
                    }
                )

        arch_notes = {
            _BASE_A: "frozen Fourier Stage A; single-light SV eval",
            _BASE_B: "frozen Fourier Stage A; xyz mean over lights",
            f"{_BASE_A}_pooled": "frozen pooled Stage A; single-light SV eval",
            f"{_BASE_B}_pooled": "frozen pooled Stage A; xyz mean over lights",
        }
        epochs_b_map = {
            _BASE_A: float("nan"),
            _BASE_B: float("nan"),
            head_c.variant_id: len(fit_c["history"]),
            head_d.variant_id: len(fit_d["history"]),
            f"{_BASE_A}_pooled": float("nan"),
            f"{_BASE_B}_pooled": float("nan"),
            head_c_p.variant_id: (
                len(fit_c_p["history"]) if fit_c_p is not None else float("nan")
            ),
            head_d_p.variant_id: (
                len(fit_d_p["history"]) if fit_d_p is not None else float("nan")
            ),
        }
        lr_a_map = {"fourier": float(lr_fourier), "pooled": float(lr_pooled_a)}
        lr_b_map = dict(lr_map)
        epochs_a_map = {"fourier": float(epochs_a_f), "pooled": float(epochs_a_p)}

        for variant, meta in metrics_by_variant.items():
            by = meta["by_split"]
            desc = meta["describe"]
            if desc is None:
                arch_json = ""
                arch_note = arch_notes.get(variant, "")
                n_params = None
                freeze_enc = cfg.mode == "frozen"
                end_to_end = cfg.mode == "e2e"
            else:
                arch_json = json.dumps(desc, default=str)
                arch_note = str(desc.get("note", ""))
                n_params = desc.get("learned_parameter_count")
                freeze_enc = bool(
                    desc.get("freeze_encoder", cfg.mode == "frozen")
                )
                end_to_end = bool(desc.get("end_to_end", cfg.mode == "e2e"))
            kind = meta["backbone_kind"]
            session_history_rows.append(
                build_history_row(
                    run_id=run_id,
                    repeat_index=rep,
                    timestamp_utc=ts,
                    notebook_id=names.notebook_id,
                    experiment_id=names.experiment_id,
                    variant_id=variant,
                    fusion_pattern=meta["fusion_pattern"],
                    backbone_kind=kind,
                    metrics_by_split=by,
                    architecture_note=arch_note,
                    architecture_json=arch_json,
                    freeze_encoder=freeze_enc,
                    end_to_end=end_to_end,
                    learned_parameter_count=n_params,
                    n_views=meta["n_views"],
                    fixed_camera_deg=cfg.fixed_camera_deg,
                    single_light_ref_deg=cfg.single_light_ref_deg,
                    light_geometry=meta["light_geometry"],
                    lr_stage_a=lr_a_map.get(kind, float("nan")),
                    lr_stage_b=lr_b_map.get(variant, float("nan")),
                    epochs_max=int(cfg.num_epochs),
                    early_stop_patience=int(cfg.early_stop_patience),
                    epochs_ran_stage_a=epochs_a_map.get(kind, float("nan")),
                    epochs_ran_stage_b=epochs_b_map.get(variant, float("nan")),
                    batch_size=int(cfg.batch_size),
                    quick=False,
                    run_lr_study=bool(do_lr_study),
                    include_pooled_control=bool(cfg.include_pooled_control),
                    n_repeat_training_requested=int(cfg.n_repeat_training),
                    n_repeat_training_effective=int(n_rep),
                    seed="",
                )
            )

        last_comparison_rows = comparison_rows
        last_payload = {
            "stage_a_fourier": stage_a_state,
            "stage_a_pooled": stage_a_state_p,
            "c": clone_state_dict(model_c),
            "d": clone_state_dict(model_d),
            "c_pooled": None if model_c_p is None else clone_state_dict(model_c_p),
            "d_pooled": None if model_d_p is None else clone_state_dict(model_d_p),
            "describe_c": model_c.describe(),
            "describe_d": model_d.describe(),
            "describe_c_pooled": None if model_c_p is None else model_c_p.describe(),
            "describe_d_pooled": None if model_d_p is None else model_d_p.describe(),
            "selected_lr_c": selected_lr_c,
            "selected_lr_d": selected_lr_d,
            "selected_lr_c_p": selected_lr_c_p,
            "selected_lr_d_p": selected_lr_d_p,
        }

    run_history_df = append_run_history(history_path, session_history_rows)
    if cfg.verbose:
        print(
            f"Appended {len(session_history_rows)} rows → "
            f"{repo_relative_path(history_path)}  "
            f"(run_ids={session_run_ids}; history_rows={len(run_history_df)})",
            flush=True,
        )

    recommended: dict[str, Any] = {
        "LR_STAGE_B_C_FOURIER": last_payload["selected_lr_c"],
        "LR_STAGE_B_D_FOURIER": last_payload["selected_lr_d"],
        "LR_STAGE_B_C_POOLED": last_payload["selected_lr_c_p"],
        "LR_STAGE_B_D_POOLED": last_payload["selected_lr_d_p"],
        "LR_STAGE_A_POOLED": lr_pooled_a,
        "run_lr_study": cfg.run_lr_study,
        "select_best_val_lr": cfg.select_best_val_lr,
        "n_repeat_training": n_rep,
        "session_run_ids": session_run_ids,
    }
    if cfg.mode == "frozen":
        recommended.update(
            {
                "use_angle_film": bool(cfg.use_angle_film),
                "geometry_mode": geometry_mode_d,
                "effective_run_lr_study": effective_run_lr_study,
                "note": (
                    "With RUN_LR_STUDY='if_unknown', compatible LRs in this JSON are "
                    "reused automatically; D LRs require matching use_angle_film. "
                    "Set RUN_LR_STUDY=False to force no sweep, or True to resweep all."
                ),
            }
        )
    else:
        recommended["note"] = (
            "Pass 2: paste LR_STAGE_B_C/D_FOURIER and LR_STAGE_B_C/D_POOLED "
            "into config, set run_lr_study=False, increase n_repeat_training, "
            "re-run for final eval + history."
        )

    lr_study_df = persist_stage_b_lr_artifacts(
        results_dir,
        lr_study_fourier_df=last_lr_study_fourier_df,
        lr_study_pooled_df=last_lr_study_pooled_df,
        include_pooled_control=cfg.include_pooled_control,
        recommended=recommended,
        csv_lr_study=names.csv_lr_study,
        csv_lr_study_fourier=names.csv_lr_study_fourier,
        csv_lr_study_pooled=names.csv_lr_study_pooled,
        json_recommended_lrs=names.json_recommended_lrs,
    )

    comparison_df = pd.DataFrame(last_comparison_rows or [])
    comparison_path = results_dir / names.csv_comparison
    comparison_df.to_csv(comparison_path, index=False)

    session_summary_df = aggregate_run_history(pd.DataFrame(session_history_rows))
    session_summary_path = results_dir / names.csv_session_summary
    session_summary_df.to_csv(session_summary_path, index=False)

    json_payload = {
        "experiment_id": names.experiment_id,
        "mode": cfg.mode,
        "fixed_camera_deg": cfg.fixed_camera_deg,
        "light_angles_deg": list(light_angles),
        "single_light_ref_deg": cfg.single_light_ref_deg,
        "include_pooled_control": cfg.include_pooled_control,
        "run_lr_study": cfg.run_lr_study,
        "select_best_val_lr": cfg.select_best_val_lr,
        "n_repeat_training": n_rep,
        "session_run_ids": session_run_ids,
        "selected_lrs": recommended,
        "lr_study": lr_study_df.to_dict(orient="records"),
        "c": last_payload["describe_c"],
        "d": last_payload["describe_d"],
        "c_pooled": last_payload["describe_c_pooled"],
        "d_pooled": last_payload["describe_d_pooled"],
        "comparison_last_run": comparison_df.to_dict(orient="records"),
        "session_summary": session_summary_df.to_dict(orient="records"),
        "run_history_path": str(repo_relative_path(history_path)),
    }
    if cfg.mode == "frozen":
        json_payload.update(
            {
                "use_angle_film": bool(cfg.use_angle_film),
                "geometry_mode": geometry_mode_d,
                "effective_run_lr_study": effective_run_lr_study,
            }
        )
    json_path = results_dir / names.json_name
    json_path.write_text(json.dumps(json_payload, indent=2, default=str))

    checkpoint_blob = {
        **last_payload,
        "mode": cfg.mode,
        "experiment_id": names.experiment_id,
        "session_run_ids": session_run_ids,
        "selected_lrs": recommended,
        "comparison_last_run": comparison_df.to_dict(orient="records"),
        "session_summary": session_summary_df.to_dict(orient="records"),
        "lr_study": lr_study_df.to_dict(orient="records"),
    }
    save_study_checkpoint(checkpoint_path, checkpoint_blob)

    if cfg.verbose:
        print(f"Wrote {repo_relative_path(comparison_path)}")
        print(f"Wrote {repo_relative_path(session_summary_path)}")
        print(f"Wrote {repo_relative_path(json_path)}")
        print(f"Wrote {repo_relative_path(checkpoint_path)}")

    return M10IlluminationResult(
        mode=cfg.mode,
        comparison_df=comparison_df,
        session_summary_df=session_summary_df,
        lr_study_df=lr_study_df,
        run_history_df=run_history_df,
        results_dir=results_dir,
        comparison_path=comparison_path,
        checkpoint_path=checkpoint_path,
        session_run_ids=session_run_ids,
        skipped_train=False,
        selected_lrs=recommended,
        session_summary_path=session_summary_path,
        history_path=history_path,
        lr_study_path=results_dir / names.csv_lr_study,
        json_path=json_path,
        extra={
            "experiment_id": names.experiment_id,
            "effective_run_lr_study": effective_run_lr_study,
            "geometry_mode": geometry_mode_d if cfg.mode == "frozen" else None,
        },
    )
