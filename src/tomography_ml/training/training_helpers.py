"""Shared training helpers for single-view and fusion localisation.

Provides catalog ``batch_xy`` factories, end-to-end (e2e) fusion training with
validation root-mean-square error (RMSE) early stopping, learning-rate study
orchestration, and illumination-only dataset builders. Single-view full-split
training lives in
:func:`tomography_ml.localization.architecture_capability.train_full_split`.

Train/validation/test splits are always by ``sequence_id`` — helpers assume
callers built datasets with the correct split filters.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from gummybear.paths import repo_relative_path
from tomography_ml.gummybear_data_catalog.catalog import CatalogRow
from tomography_ml.gummybear_data_catalog.task_dataset import (
    DatasetTaskSpec,
    build_task_dataset,
)
from tomography_ml.localization.architecture_capability import (
    mse_loss,
    per_axis_rmse,
)

PredictFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MakeModelFn = Callable[[], nn.Module]
BatchXyFn = Callable[
    [Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
    tuple[torch.Tensor, torch.Tensor],
]


def lr_close(a: float, b: float) -> bool:
    """Test whether two learning rates match for learning rate (LR) selection.

    Uses a relative tolerance scaled by magnitude so grid values like
    ``3e-2`` and ``0.03`` compare equal when selecting cached checkpoints.

    Args:
        a: First learning rate.
        b: Second learning rate.

    Returns:
        ``True`` when ``|a - b| ≤ 1e-12 * max(1, |a|, |b|)``.
    """
    a_f, b_f = float(a), float(b)
    return abs(a_f - b_f) <= 1e-12 * max(1.0, abs(a_f), abs(b_f))


def batch_from_indices(
    ds: Dataset,
    indices: Sequence[int],
    device: torch.device | str,
    y_fields: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack multi-illumination dataset rows for manual minibatch training.

    Expects each ``ds[i]`` to return ``(views, targets_dict, light_angles)``
    as produced by multi-illumination stack datasets.

    Args:
        ds: Dataset with per-item views, target dict, and light angles.
        indices: Row indices to stack into one minibatch.
        device: Target device for returned tensors.
        y_fields: Target keys to stack into ``[B, len(y_fields)]``.

    Returns:
        ``(views, targets, light_angles)`` with shapes ``[B, …]``,
        ``[B, n_out]`` float32, and light angles on ``device``.
    """
    views_list: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    lights: list[torch.Tensor] = []
    for i in indices:
        views, targets, light = ds[int(i)]
        views_list.append(views)
        tgts.append(torch.tensor([float(targets[n]) for n in y_fields]))
        lights.append(light)
    return (
        torch.stack(views_list, dim=0).to(device=device),
        torch.stack(tgts, dim=0).to(device=device, dtype=torch.float32),
        torch.stack(lights, dim=0).to(device=device),
    )


def make_batch_xy_single(
    *,
    x_field: str,
    y_fields: Sequence[str],
    device: torch.device | str,
) -> BatchXyFn:
    """Build a ``batch_xy`` callable for single-view catalog DataLoaders.

    Accepts ``(images, targets)`` dict batches (as from ``CatalogTaskDataset``),
    squeezes a leading view axis when present (``[B,1,C,H,W] → [B,C,H,W]``),
    and stacks ``y_fields`` into ``[B, n_out]``.

    Args:
        x_field: Catalog image key (e.g. ``anomaly_ref``).
        y_fields: Target keys in column order.
        device: Device for returned tensors.

    Returns:
        Callable ``batch_xy(images, targets) -> (views [B,C,H,W], y [B,n_out])``.

    Typically used with :class:`~tomography_ml.gummybear_data_catalog.task_dataset.CatalogTaskDataset`
    and single-view models such as :class:`~tomography_ml.localization.localizer.LocalizerSingleViewFourier`.
    """

    y_names = tuple(str(n) for n in y_fields)
    field = str(x_field)

    def batch_xy(
        images: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        views = images[field]
        if views.ndim == 5:
            views = views[:, 0]
        views = views.to(device=device, dtype=torch.float32)
        batch_targets = torch.stack([targets[n] for n in y_names], dim=1).to(
            device=device, dtype=torch.float32
        )
        return views, batch_targets

    return batch_xy


def make_batch_xy_multiview(
    *,
    x_field: str,
    y_fields: Sequence[str],
    device: torch.device | str,
) -> BatchXyFn:
    """Build a ``batch_xy`` callable that keeps the full view stack.

    Unlike :func:`make_batch_xy_single`, does **not** drop the view axis.
    Ensures ``views`` has shape ``[B, V, C, H, W]`` (unsqueezes a missing V
    when the loader yields ``[B, C, H, W]``).

    Args:
        x_field: Catalog image key for the multi-view stack.
        y_fields: Target keys in column order.
        device: Device for returned tensors.

    Returns:
        Callable ``batch_xy(images, targets) -> (views [B,V,C,H,W], y [B,n_out])``.

    Typically used with :class:`~tomography_ml.gummybear_data_catalog.task_dataset.CatalogTaskDataset`
    and multi-view fusion models in :mod:`tomography_ml.localization.localize_multiview`.
    """

    y_names = tuple(str(n) for n in y_fields)
    field = str(x_field)

    def batch_xy(
        images: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        views = images[field].to(device=device, dtype=torch.float32)
        if views.ndim == 4:
            views = views.unsqueeze(1)
        batch_targets = torch.stack([targets[n] for n in y_names], dim=1).to(
            device=device, dtype=torch.float32
        )
        return views, batch_targets

    return batch_xy


def collect_prediction_errors(
    model: nn.Module,
    loader: Iterable[tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]],
    batch_xy: BatchXyFn,
) -> tuple[np.ndarray, float]:
    """Collect per-sample Euclidean (L2) errors and mean squared error (MSE) over a single-view loader.

    Args:
        model: Trained single-view localiser (eval mode assumed).
        loader: Iterable of ``(images, targets)`` mapping batches.
        batch_xy: Typically from :func:`make_batch_xy_single`.

    Returns:
        ``(sample_l2_errors, mean_mse)`` where ``sample_l2_errors`` is
        ``np.ndarray`` shape ``[N]`` (Euclidean (L2) norm per row) and
        ``mean_mse`` is the dataset-wide mean squared error (MSE). Empty
        loader → ``([], nan)``.

    Typically used with :func:`make_batch_xy_single` and frozen single-view experts.
    """
    model.eval()
    errs: list[np.ndarray] = []
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for images, targets in loader:
            views, batch_targets = batch_xy(images, targets)
            pred = model(views)
            diff = pred - batch_targets
            total_mse += float((diff**2).mean().item()) * int(views.shape[0])
            count += int(views.shape[0])
            errs.append(
                torch.linalg.norm(diff, dim=1).detach().cpu().numpy()
            )
    if not errs:
        return np.asarray([], dtype=np.float64), float("nan")
    return np.concatenate(errs), total_mse / max(count, 1)


def train_e2e(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    y_fields: Sequence[str],
    device: torch.device | str,
    use_angles: bool,
    lr: float,
    num_epochs: int,
    early_stop_patience: int,
    batch_size: int,
    progress_label: str = "Stage B",
) -> dict[str, Any]:
    """Train a multi-view fusion model with validation root-mean-square error (RMSE) early stopping.

Manual epoch loop over ``train_ds`` with shuffled indices; validation RMSE uses
:func:`~tomography_ml.localization.architecture_capability.per_axis_rmse`
on ``val_ds``. Best validation weights are restored at the end.

    Expects dataset items ``(views, targets_dict, light_angles)``. When
    ``use_angles`` is true, calls ``model(views, angles_deg=lights)``.

    Args:
        model: Fusion localiser (trainable parameters only are optimised).
        train_ds: Training stack dataset (train split by ``sequence_id``).
        val_ds: Validation stack dataset (never test).
        y_fields: Target keys stacked to ``[B, n_out]``.
        device: Torch device.
        use_angles: Pass ``light_angles`` into the model when ``True``.
        lr: Adam learning rate (LR).
        num_epochs: Maximum epoch budget.
        early_stop_patience: Stop after this many epochs without val improvement.
        batch_size: Minibatch size for train and val passes.
        progress_label: Prefix for per-epoch log lines.

    Returns:
        ``{"history": [...], "best_val_rmse": float}`` where each history row
        has ``epoch``, ``train_loss``, and per-axis RMSE keys from
        :func:`per_axis_rmse`.

    Typically used with :class:`~tomography_ml.gummybear_data_catalog.IlluminationOnlyDataset.IlluminationOnlyDataset`
    and fusion models; log results via :func:`~tomography_ml_validation.run_history.build_history_row`.
    """
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr),
    )
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stall = 0
    history: list[dict[str, Any]] = []
    n_train = len(train_ds)
    label = str(progress_label).strip() or "Stage B"
    y_names = tuple(str(n) for n in y_fields)
    print(
        f"=== {label}  lr={float(lr):g}  epochs≤{int(num_epochs)}  "
        f"early_stop={int(early_stop_patience)} ===",
        flush=True,
    )
    for epoch in range(int(num_epochs)):
        model.train()
        order = torch.randperm(n_train)
        total = 0.0
        count = 0
        for start in range(0, n_train, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            views, y, lights = batch_from_indices(train_ds, idx, device, y_names)
            pred = (
                model(views, angles_deg=lights) if use_angles else model(views)
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
            for start in range(0, len(val_ds), int(batch_size)):
                idx = list(range(start, min(start + int(batch_size), len(val_ds))))
                views, y, lights = batch_from_indices(val_ds, idx, device, y_names)
                pred = (
                    model(views, angles_deg=lights) if use_angles else model(views)
                )
                preds.append(pred.cpu())
                tgts.append(y.cpu())
        metrics = per_axis_rmse(torch.cat(preds), torch.cat(tgts), y_names)
        val_rmse = metrics["train_RMSE_total"]
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        improved = False
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            stall = 0
            improved = True
        else:
            stall += 1
        marker = "*" if improved else " "
        print(
            f"  {label} epoch {epoch + 1:03d}/{int(num_epochs)}  "
            f"train_loss={train_loss:.5f}  "
            f"val_RMSE={val_rmse:.4f}  best={best_val:.4f}{marker}  "
            f"stall={stall}/{int(early_stop_patience)}",
            flush=True,
        )
        if stall >= int(early_stop_patience):
            print(f"  {label} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(
        f"  {label} done  best_val={best_val:.4f}  epochs={len(history)}",
        flush=True,
    )
    return {"history": history, "best_val_rmse": best_val}


def run_stage_b_lr_study(
    *,
    tag: str,
    make_model: MakeModelFn,
    train_ds: Dataset,
    val_ds: Dataset,
    y_fields: Sequence[str],
    device: torch.device | str,
    use_angles: bool,
    lr_fixed: float,
    run_study: bool,
    select_best_val_lr: bool,
    lr_grid: Sequence[float],
    num_epochs: int,
    early_stop_patience: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run a learning-rate sweep or a single fixed learning rate (LR) fusion training run.

    When ``run_study`` is true, trains one fresh model per entry in
    ``lr_grid`` (always includes ``lr_fixed`` if missing from the grid).
    Selects the checkpoint by lowest validation root-mean-square error (RMSE)
    when ``select_best_val_lr`` and ``run_study`` are both true; otherwise
    keeps ``lr_fixed``.

    Args:
        tag: Variant label for logs and the study table ``variant_tag`` column.
        make_model: Factory returning a new fusion model per LR candidate.
        train_ds / val_ds: Stack datasets for the train and validation splits.
        y_fields: Target column order.
        device: Torch device.
        use_angles: Forward with ``angles_deg=lights`` when true.
        lr_fixed: Default learning rate (LR) when not sweeping or when
            selection is disabled.
        run_study: If false, train only at ``lr_fixed``.
        select_best_val_lr: Pick lowest ``best_val_rmse`` from the sweep.
        lr_grid: Candidate learning rates for the sweep.
        num_epochs / early_stop_patience / batch_size: Passed to :func:`train_e2e`.

    Returns:
        Dict with keys ``model`` (eval, loaded with selected weights),
        ``fit``, ``describe``, ``selected_lr``, ``selection_mode``
        (``best_val_rmse`` or ``fixed_lr``), ``study_df`` (one row per LR),
        ``study_best_lr``, and ``study_best_val``.

    Persists selected LRs via :func:`persist_stage_b_lr_artifacts`; pairs with
    :func:`~tomography_ml_validation.run_history.build_history_row` in notebooks.
    """
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
        print(
            f"=== Stage B {tag}  lr={lr_b:g}  "
            f"({i + 1}/{len(lr_candidates)}) ===",
            flush=True,
        )
        model_b = make_model()
        if i == 0:
            print(model_b.describe())
        fit_b = train_e2e(
            model_b,
            train_ds,
            val_ds,
            y_fields=y_fields,
            device=device,
            use_angles=use_angles,
            lr=lr_b,
            num_epochs=num_epochs,
            early_stop_patience=early_stop_patience,
            batch_size=batch_size,
            progress_label=f"{tag} lr={lr_b:g}",
        )
        best_val = float(fit_b["best_val_rmse"])
        final_train = (
            float(fit_b["history"][-1]["train_loss"])
            if fit_b["history"]
            else float("nan")
        )
        lr_rows.append(
            {
                "lr": lr_b,
                "variant_tag": tag,
                "use_angles": bool(use_angles),
                "best_val_rmse": best_val,
                "epochs_ran": len(fit_b["history"]),
                "final_train_loss": final_train,
                "converged_hint": bool(best_val < 2.0 and final_train < 10.0),
                "trainable": model_b.learned_parameter_count(),
            }
        )
        states_by_lr[lr_b] = {
            k: v.detach().cpu().clone() for k, v in model_b.state_dict().items()
        }
        fits_by_lr[lr_b] = fit_b
        describes_by_lr[lr_b] = model_b.describe()
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
    for row in lr_rows:
        row["used_for_eval"] = lr_close(row["lr"], selected_lr)
    study_df = pd.DataFrame(lr_rows).sort_values("best_val_rmse")
    print(
        f"Selected {tag}: lr={selected_lr:g}  "
        f"best_val={float(fit['best_val_rmse']):.4f}  mode={selection}",
        flush=True,
    )
    return {
        "model": model,
        "fit": fit,
        "describe": describes_by_lr[selected_key],
        "selected_lr": selected_lr,
        "selection_mode": selection,
        "study_df": study_df,
        "study_best_lr": study_best_lr,
        "study_best_val": study_best_val,
    }


def make_sv_illumination_dataset(
    catalog_rows: Sequence[CatalogRow],
    *,
    split: str,
    light_angle_deg: float,
    x_field: str,
    y_fields: Sequence[str],
    fixed_camera_deg: float,
    image_normalize: str,
    name_prefix: str = "m10_1_sv",
):
    """Build a single-view catalog dataset for one illumination angle.

    Fixed camera at ``fixed_camera_deg``, one row per ``light_angle_deg`` via
    ``opt_m10_illum_{angle:03d}`` filter. Used for single-light baselines and
    frozen-expert calibration before multi-light fusion training.

    Args:
        catalog_rows: Full or split-filtered catalog table.
        split: ``train``, ``validation``, or ``test`` (by ``sequence_id``).
        light_angle_deg: Illumination angle in degrees (catalog optical setup).
        x_field: Input image field (e.g. ``anomaly_ref``).
        y_fields: Target fields for localisation.
        fixed_camera_deg: ``keep_angles_deg`` — single retained camera view.
        image_normalize: Preprocess normalisation mode.
        name_prefix: Prefix for ``DatasetTaskSpec.name``.

    Returns:
        A :class:`~tomography_ml.gummybear_data_catalog.task_dataset.CatalogTaskDataset`
        (lazy tensor dataset) for the requested split and illumination.

    Notebook: ``10_1A`` / ``10_1B`` Stage A.
    """
    oid = f"opt_m10_illum_{int(light_angle_deg):03d}"
    task = DatasetTaskSpec(
        name=f"{name_prefix}_{split}_{oid}",
        row_filter={
            "field_status": "complete",
            "split": split,
            "optical_setup_id": oid,
        },
        x_fields=(x_field,),
        y_fields=tuple(str(n) for n in y_fields),
        keep_angles_deg=float(fixed_camera_deg),
        image_normalize=image_normalize,
    )
    return build_task_dataset(catalog_rows, task)


def eval_stack(
    ds: Dataset,
    predict_fn: PredictFn,
    *,
    y_fields: Sequence[str],
    device: torch.device | str,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate root-mean-square error (RMSE) on a multi-view illumination stack dataset.

    Iterates the dataset in ``batch_size`` chunks and delegates metric
    computation to :func:`per_axis_rmse`.

    Args:
        ds: Stack dataset yielding ``(views, targets, light_angles)``.
        predict_fn: ``(views, lights) -> pred`` with ``pred`` shape ``[B, n_out]``.
        y_fields: Target column order.
        device: Torch device for batching.
        batch_size: Chunk size for evaluation passes.

    Returns:
        Dict with ``train_RMSE_total``, ``train_RMSE_X/Y/Z``, and per-column
        keys from :func:`per_axis_rmse` (prefix ``train_`` is historical).
    """
    preds: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    y_names = tuple(str(n) for n in y_fields)
    with torch.no_grad():
        for start in range(0, len(ds), int(batch_size)):
            idx = list(range(start, min(start + int(batch_size), len(ds))))
            views, y, lights = batch_from_indices(ds, idx, device, y_names)
            preds.append(predict_fn(views, lights).cpu())
            tgts.append(y.cpu())
    return per_axis_rmse(torch.cat(preds), torch.cat(tgts), y_names)


def persist_stage_b_lr_artifacts(
    results_dir: Path | str,
    *,
    lr_study_fourier_df: pd.DataFrame,
    lr_study_pooled_df: pd.DataFrame,
    include_pooled_control: bool,
    recommended: Mapping[str, Any],
    csv_lr_study: str,
    csv_lr_study_fourier: str,
    csv_lr_study_pooled: str,
    json_recommended_lrs: str,
) -> pd.DataFrame:
    """Write fusion learning rate (LR) study CSV tables and recommended-LR JSON file to ``results_dir``.

    Persists the combined Fourier (+ optional pooled) study table, per-variant
    CSV tables, and a JSON fingerprint consumed by :func:`load_recommended_stage_b_lrs`
    and :func:`resolve_run_lr_study`. Prints suggested constant assignments for
    the four fusion-head learning rates.

    Args:
        results_dir: Output directory (created if missing).
        lr_study_fourier_df: Fourier head sweep rows.
        lr_study_pooled_df: Pooled control sweep rows (may be empty).
        include_pooled_control: Write pooled CSV table and print pooled LR constants.
        recommended: Mapping with keys like ``LR_STAGE_B_C_FOURIER`` and
            optionally ``use_angle_film`` for D-head fingerprinting.
        csv_lr_study / csv_lr_study_fourier / csv_lr_study_pooled: Filenames.
        json_recommended_lrs: Filename for the recommended-LR JSON sidecar.

    Returns:
        Combined study :class:`pandas.DataFrame` (Fourier + optional pooled).

    Notebook: ``10_1A`` / ``10_1B`` Stage B artifact persistence.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    lr_parts = [lr_study_fourier_df]
    if include_pooled_control and len(lr_study_pooled_df):
        lr_parts.append(lr_study_pooled_df)
    lr_study_df = pd.concat(lr_parts, ignore_index=True)
    lr_study_df.to_csv(results_dir / csv_lr_study, index=False)
    lr_study_fourier_df.to_csv(results_dir / csv_lr_study_fourier, index=False)
    if include_pooled_control and len(lr_study_pooled_df):
        lr_study_pooled_df.to_csv(results_dir / csv_lr_study_pooled, index=False)
    (results_dir / json_recommended_lrs).write_text(
        json.dumps(dict(recommended), indent=2, default=str)
    )
    print(f"Wrote {repo_relative_path(results_dir / csv_lr_study)}")
    print(flush=True)
    print("=== Pass-2 paste targets (all four Stage B heads) ===", flush=True)
    print(
        f"LR_STAGE_B_C_FOURIER = {recommended['LR_STAGE_B_C_FOURIER']:g}",
        flush=True,
    )
    print(
        f"LR_STAGE_B_D_FOURIER = {recommended['LR_STAGE_B_D_FOURIER']:g}",
        flush=True,
    )
    if (
        include_pooled_control
        and recommended.get("LR_STAGE_B_C_POOLED") is not None
    ):
        print(f"Wrote {repo_relative_path(results_dir / csv_lr_study_pooled)}")
        print(
            f"LR_STAGE_B_C_POOLED = {recommended['LR_STAGE_B_C_POOLED']:g}",
            flush=True,
        )
        print(
            f"LR_STAGE_B_D_POOLED = {recommended['LR_STAGE_B_D_POOLED']:g}",
            flush=True,
        )
    print("RUN_LR_STUDY = False  # or \"if_unknown\" to skip when known", flush=True)
    print(f"Wrote {repo_relative_path(results_dir / json_recommended_lrs)}")
    return lr_study_df


# Stage B LR keys written by 10_1A / 10_1B notebooks.
STAGE_B_LR_KEYS = (
    "LR_STAGE_B_C_FOURIER",
    "LR_STAGE_B_D_FOURIER",
    "LR_STAGE_B_C_POOLED",
    "LR_STAGE_B_D_POOLED",
)
# Angle-conditioned (D) heads; fingerprint must match use_angle_film.
STAGE_B_ANGLE_LR_KEYS = (
    "LR_STAGE_B_D_FOURIER",
    "LR_STAGE_B_D_POOLED",
)


def load_recommended_stage_b_lrs(
    path: Path | str,
) -> dict[str, Any] | None:
    """Load a persisted recommended learning rate (LR) JSON sidecar.

    Typical path: ``…/m10_1a_recommended_lrs.json`` written by
    :func:`persist_stage_b_lr_artifacts`. Used by ``RUN_LR_STUDY = "if_unknown"``
    to skip sweeps when the fingerprint matches.

    Args:
        path: JSON file path.

    Returns:
        Parsed dict on success; ``None`` if missing, unreadable, or not a dict.

    Notebook: ``10_1A`` / ``10_1B``.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def stage_b_lr_key_is_known(
    recommended: Mapping[str, Any] | None,
    key: str,
    *,
    use_angle_film: bool,
) -> bool:
    """Test whether a recommended learning rate (LR) key is valid for the current fusion geometry.

    Angle-conditioned D heads (``LR_STAGE_B_D_*``) require a matching
    ``use_angle_film`` flag in the JSON file. Legacy JSON without that key is
    treated as ``use_angle_film=False`` only.

    Args:
        recommended: Loaded JSON from :func:`load_recommended_stage_b_lrs`, or
            ``None``.
        key: One of :data:`STAGE_B_LR_KEYS` / :data:`STAGE_B_ANGLE_LR_KEYS`.
        use_angle_film: Whether the current run uses Feature-wise Linear
            Modulation (FiLM) angle conditioning.

    Returns:
        ``True`` when ``key`` holds a numeric LR compatible with the current
        D-head geometry fingerprint.
    """
    if recommended is None:
        return False
    if key not in recommended or recommended[key] is None:
        return False
    try:
        float(recommended[key])
    except (TypeError, ValueError):
        return False
    if key in STAGE_B_ANGLE_LR_KEYS:
        stored = recommended.get("use_angle_film")
        if stored is None:
            # Legacy concat-era JSON (pre-FiLM flag) only matches film=False.
            return not bool(use_angle_film)
        return bool(stored) == bool(use_angle_film)
    return True


def resolve_run_lr_study(
    run_lr_study: bool | str,
    *,
    recommended_path: Path | str,
    use_angle_film: bool,
    include_pooled_control: bool,
    lr_defaults: Mapping[str, float],
) -> dict[str, Any]:
    """Resolve fusion learning rate (LR) study flags and effective learning rates.

    ``run_lr_study`` modes:
      - ``True`` — always sweep all required heads
      - ``False`` — never sweep; use ``lr_defaults`` (and known JSON overrides)
      - ``\"if_unknown\"`` — sweep only heads whose recommended LR is missing
        or incompatible with ``use_angle_film`` (auto-load from JSON when valid)

    Required keys depend on ``include_pooled_control`` (C/D Fourier always;
    C/D pooled optional). Splits are assumed correct on the caller's datasets
    (train/val/test by ``sequence_id``).

    Args:
        run_lr_study: Boolean or ``\"if_unknown\"`` / aliases (see implementation).
        recommended_path: Path to recommended-LR JSON sidecar.
        use_angle_film: Current D-head Feature-wise Linear Modulation (FiLM)
            geometry (fingerprint check).
        include_pooled_control: Whether pooled LR keys are required.
        lr_defaults: Fallback LRs for keys not loaded from JSON.

    Returns:
        Dict with ``mode`` (``always`` | ``never`` | ``if_unknown``),
        ``effective_run_lr_study``, ``study_by_key`` (per-head bool),
        ``lrs`` (effective floats), ``loaded_recommended``, ``known_keys``,
        ``unknown_keys``, and ``use_angle_film``.
    """
    mode = run_lr_study
    if isinstance(mode, str):
        mode_key = mode.strip().lower()
        if mode_key in ("if_unknown", "unknown", "auto"):
            mode = "if_unknown"
        elif mode_key in ("true", "1", "yes"):
            mode = True
        elif mode_key in ("false", "0", "no"):
            mode = False
        else:
            raise ValueError(
                "run_lr_study must be True, False, or 'if_unknown'; "
                f"got {run_lr_study!r}"
            )

    required = list(STAGE_B_LR_KEYS[:2])
    if include_pooled_control:
        required.extend(STAGE_B_LR_KEYS[2:])

    loaded = load_recommended_stage_b_lrs(recommended_path)
    known: list[str] = []
    unknown: list[str] = []
    for key in required:
        if stage_b_lr_key_is_known(
            loaded, key, use_angle_film=use_angle_film
        ):
            known.append(key)
        else:
            unknown.append(key)

    lrs = {k: float(lr_defaults[k]) for k in required}
    if loaded is not None:
        for key in known:
            lrs[key] = float(loaded[key])

    if mode is True:
        study_by_key = {k: True for k in required}
        effective = True
    elif mode is False:
        study_by_key = {k: False for k in required}
        effective = False
    else:
        study_by_key = {k: (k in unknown) for k in required}
        effective = bool(unknown)

    return {
        "mode": (
            "always"
            if mode is True
            else ("never" if mode is False else "if_unknown")
        ),
        "effective_run_lr_study": effective,
        "study_by_key": study_by_key,
        "lrs": lrs,
        "loaded_recommended": loaded,
        "known_keys": known,
        "unknown_keys": unknown,
        "use_angle_film": bool(use_angle_film),
    }