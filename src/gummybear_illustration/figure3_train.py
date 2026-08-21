"""Controlled M8 pooling vs Fourier training with per-epoch prediction logs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gummybear.paths import display_path
from tomography_ml.gummybear_data_catalog.catalog import CatalogRow, build_catalog_rows
from tomography_ml.gummybear_data_catalog.gummybear_adapter import load_catalog_jobs
from tomography_ml.gummybear_data_catalog.task_dataset import (
    DatasetTaskSpec,
    build_task_dataset,
)
from tomography_ml.localization.builders import materialize_lazy_modules
from tomography_ml.studies.single_view_m8 import (
    M8_CANONICAL_LR_BY_ARCH,
    make_batch_xy_single,
    make_m8_single_view_model,
)
from tomography_ml.training.training_helpers import collect_prediction_errors

from .figure3_history import (
    MODEL_FOURIER,
    MODEL_POOLING,
    append_record,
    combine_histories,
    default_figure3_root,
    empty_history,
    history_paths,
    save_history,
)
from .paths import default_m8_data_root, repo_root

Y_FIELDS: tuple[str, ...] = ("particle_x", "particle_y", "particle_z")
X_FIELD = "anomaly_ref"
OPTICAL_SETUP_ID = "opt_m8_high_001"
KEEP_ANGLE_DEG = 180.0
ARCH_POOLING = "pooled"
ARCH_FOURIER = "fourier"


def _split_task(task: DatasetTaskSpec, *, name: str, split: str) -> DatasetTaskSpec:
    return replace(
        task,
        name=name,
        row_filter={**dict(task.row_filter), "split": split},
    )


def m8_xyz_task(*, split: str = "train") -> DatasetTaskSpec:
    """Match the final-report M8 xyz single-view protocol (180°, z-score)."""
    return DatasetTaskSpec(
        name=f"figure3_localization_xyz_{split}",
        row_filter={
            "split": split,
            "field_status": "complete",
            "optical_setup_id": OPTICAL_SETUP_ID,
        },
        x_fields=(X_FIELD,),
        y_fields=Y_FIELDS,
        keep_angles_deg=KEEP_ANGLE_DEG,
        image_normalize="per_image_zscore",
    )


def load_m8_catalog_rows(
    *,
    repo: Path | None = None,
    workbook_path: Path | None = None,
    data_root: Path | None = None,
) -> list[CatalogRow]:
    root = repo_root() if repo is None else Path(repo)
    wb = (
        Path(workbook_path)
        if workbook_path is not None
        else root / "configs" / "m8" / "localization_single_particle.xlsx"
    )
    data = (
        Path(data_root) if data_root is not None else default_m8_data_root(root)
    )
    jobs = load_catalog_jobs(wb, data, stl_root=root)
    return build_catalog_rows(jobs)


def select_tracked_split_rows(
    catalog_rows: Sequence[CatalogRow],
    *,
    split: str,
    n_tracked: int | None,
    seed: int,
    task: DatasetTaskSpec | None = None,
) -> list[CatalogRow]:
    """Deterministic subset for one catalog split (``None`` / non-positive → all)."""
    task = m8_xyz_task(split=split) if task is None else task
    split_task = _split_task(task, name=f"figure3_{split}", split=split)
    ds = build_task_dataset(list(catalog_rows), split_task)
    rows = list(ds.rows)
    if not rows:
        return []
    if n_tracked is None or int(n_tracked) <= 0 or int(n_tracked) >= len(rows):
        return rows
    n = int(n_tracked)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(rows), size=n, replace=False)
    idx = np.sort(np.asarray(idx, dtype=int))
    return [rows[int(i)] for i in idx]


def select_tracked_validation_rows(
    catalog_rows: Sequence[CatalogRow],
    *,
    n_tracked: int | None,
    seed: int,
    task: DatasetTaskSpec | None = None,
) -> list[CatalogRow]:
    """Deterministic validation subset (``None`` / non-positive → all val rows)."""
    rows = select_tracked_split_rows(
        catalog_rows,
        split="validation",
        n_tracked=n_tracked,
        seed=seed,
        task=task,
    )
    if not rows:
        raise ValueError("no validation rows match the M8 figure-3 task filter")
    return rows


def _tracked_meta(rows: Sequence[CatalogRow]) -> list[dict[str, Any]]:
    meta: list[dict[str, Any]] = []
    for row in rows:
        meta.append(
            {
                "sample_id": str(row.sequence_id),
                "catalog_sample_id": int(row.sample_id),
                "sequence_id": str(row.sequence_id),
                "split": str(row.split),
                "manifest_path": str(row.manifest_path),
                "stl_path": None,
                "y_true": [
                    float(row.particle_x),
                    float(row.particle_y),
                    float(row.particle_z),
                ],
                "particle_radius": (
                    None
                    if row.particle_radius is None
                    else float(row.particle_radius)
                ),
            }
        )
    return meta


def _predict_tracked(
    model,
    tracked_rows: Sequence[CatalogRow],
    *,
    device,
) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    import torch

    batch_xy = make_batch_xy_single(
        x_field=X_FIELD, y_fields=Y_FIELDS, device=device
    )
    model.eval()
    out: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    with torch.no_grad():
        for row in tracked_rows:
            row_task = m8_xyz_task(split=str(row.split))
            ds = build_task_dataset([row], row_task)
            x, y = ds[0]
            images = {
                X_FIELD: torch.as_tensor(np.asarray(x[X_FIELD])).unsqueeze(0)
            }
            targets = {
                name: torch.as_tensor(
                    [float(y[name])], dtype=torch.float32
                )
                for name in Y_FIELDS
            }
            views, batch_targets = batch_xy(images, targets)
            pred = model(views)
            pred_np = pred.detach().cpu().numpy().reshape(3).astype(float)
            true_np = batch_targets.detach().cpu().numpy().reshape(3).astype(float)
            err = float(np.linalg.norm(pred_np - true_np))
            out.append((str(row.sequence_id), true_np, pred_np, err))
    return out


def _epoch_loader_mse(model, loader, batch_xy) -> float:
    _errs, mse = collect_prediction_errors(model, loader, batch_xy)
    return float(mse)


@dataclass(frozen=True)
class Figure3TrainResult:
    pooling_history: dict[str, Any]
    fourier_history: dict[str, Any]
    combined_history: dict[str, Any]
    paths: dict[str, Path]
    tracked_rows: tuple[CatalogRow, ...]


def train_figure3_convergence(
    *,
    repo: Path | None = None,
    workbook_path: Path | None = None,
    data_root: Path | None = None,
    output_root: Path | None = None,
    num_epochs: int = 200,
    batch_size: int = 32,
    seed: int = 0,
    n_tracked: int | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> Figure3TrainResult:
    """Train pooled vs Fourier once each; log tracked preds every epoch to JSON.

    Does **not** write under ``checkpoints/m8/``. Uses
    :data:`M8_CANONICAL_LR_BY_ARCH` (pooled ``0.001``, fourier ``0.03``).
    """
    import torch
    from torch.nn.functional import mse_loss
    from torch.utils.data import DataLoader

    from tomography_ml import get_device

    root = repo_root() if repo is None else Path(repo)
    out_root = default_figure3_root(root) if output_root is None else Path(output_root)
    paths = history_paths(out_root)
    paths["root"].mkdir(parents=True, exist_ok=True)

    catalog_rows = load_m8_catalog_rows(
        repo=root, workbook_path=workbook_path, data_root=data_root
    )
    train_task = m8_xyz_task(split="train")
    val_task = m8_xyz_task(split="validation")
    train_ds = build_task_dataset(catalog_rows, train_task)
    val_ds = build_task_dataset(catalog_rows, val_task)
    if len(train_ds) < 1 or len(val_ds) < 1:
        raise ValueError(
            f"need non-empty train/val; got train={len(train_ds)} val={len(val_ds)}"
        )

    tracked_train = select_tracked_split_rows(
        catalog_rows,
        split="train",
        n_tracked=n_tracked,
        seed=int(seed) + 104729,
        task=train_task,
    )
    tracked_val = select_tracked_split_rows(
        catalog_rows,
        split="validation",
        n_tracked=n_tracked,
        seed=seed,
        task=train_task,
    )
    tracked_test = select_tracked_split_rows(
        catalog_rows,
        split="test",
        n_tracked=n_tracked,
        seed=int(seed) + 7919,
        task=train_task,
    )
    if not tracked_val:
        raise ValueError("no validation rows match the M8 figure-3 task filter")
    tracked_rows = tuple(tracked_train + tracked_val + tracked_test)
    tracked_meta = _tracked_meta(tracked_rows)
    # Attach STL from sequence manifest when available (for POV later).
    for meta, row in zip(tracked_meta, tracked_rows):
        meta["manifest_path"] = str(row.manifest_path)
        meta["sequence_dir"] = str(row.sequence_dir)

    torch_device = get_device() if device is None else torch.device(device)
    batch_xy = make_batch_xy_single(
        x_field=X_FIELD, y_fields=Y_FIELDS, device=torch_device
    )

    class _MapDataset(torch.utils.data.Dataset):
        def __init__(self, inner):
            self.inner = inner

        def __len__(self) -> int:
            return len(self.inner)

        def __getitem__(self, index: int):
            return self.inner[index]

    train_loader = DataLoader(
        _MapDataset(train_ds),
        batch_size=min(int(batch_size), len(train_ds)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    val_loader = DataLoader(
        _MapDataset(val_ds),
        batch_size=min(int(batch_size), len(val_ds)),
        shuffle=False,
    )

    histories: dict[str, dict[str, Any]] = {}
    for arch, model_type in (
        (ARCH_POOLING, MODEL_POOLING),
        (ARCH_FOURIER, MODEL_FOURIER),
    ):
        lr = float(M8_CANONICAL_LR_BY_ARCH[arch])
        if verbose:
            print(
                f"=== Figure 3 train {model_type} arch={arch} lr={lr:g} "
                f"epochs={num_epochs} seed={seed} "
                f"train={len(train_ds)} val={len(val_ds)} "
                f"tracked_train={len(tracked_train)} "
                f"tracked_val={len(tracked_val)} tracked_test={len(tracked_test)} "
                f"device={torch_device} ===",
                flush=True,
            )
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        model = make_m8_single_view_model(
            arch, n_outputs=len(Y_FIELDS), device=torch_device
        )
        with torch.no_grad():
            images0, targets0 = next(iter(train_loader))
            v0, _ = batch_xy(images0, targets0)
            _ = model(v0)
        materialize_lazy_modules(model, v0[:1])
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        hist = empty_history(
            model_type=model_type,
            arch=arch,
            lr=lr,
            seed=seed,
            num_epochs=num_epochs,
            y_fields=Y_FIELDS,
            tracked_samples=tracked_meta,
            extra={
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "tracked_train_size": len(tracked_train),
                "tracked_val_size": len(tracked_val),
                "tracked_test_size": len(tracked_test),
                "optical_setup_id": OPTICAL_SETUP_ID,
                "keep_angles_deg": KEEP_ANGLE_DEG,
                "image_normalize": "per_image_zscore",
                "batch_size": int(batch_size),
                "data_root": display_path(
                    default_m8_data_root(root)
                    if data_root is None
                    else Path(data_root)
                ),
            },
        )

        def _log_epoch(epoch: int, train_loss: float | None, val_loss: float | None) -> None:
            for sid, y_true, y_pred, err in _predict_tracked(
                model, tracked_rows, device=torch_device
            ):
                append_record(
                    hist,
                    epoch=epoch,
                    sample_id=sid,
                    y_true=y_true,
                    y_pred=y_pred,
                    localization_error=err,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )

        # Epoch 0: initialization (before any optimizer step).
        train0 = _epoch_loader_mse(model, train_loader, batch_xy)
        val0 = _epoch_loader_mse(model, val_loader, batch_xy)
        _log_epoch(0, train_loss=train0, val_loss=val0)
        if verbose:
            print(
                f"  {model_type} epoch 0/{num_epochs}  "
                f"train_MSE={train0:.4f}  val_MSE={val0:.4f}  (init)",
                flush=True,
            )

        for epoch in range(1, int(num_epochs) + 1):
            model.train()
            total = 0.0
            count = 0
            for images, targets in train_loader:
                views, batch_targets = batch_xy(images, targets)
                pred = model(views)
                loss = mse_loss(pred, batch_targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * int(views.shape[0])
                count += int(views.shape[0])
            train_loss = total / max(count, 1)
            val_loss = _epoch_loader_mse(model, val_loader, batch_xy)
            _log_epoch(epoch, train_loss=train_loss, val_loss=val_loss)
            if verbose and (epoch == 1 or epoch == num_epochs or epoch % 20 == 0):
                print(
                    f"  {model_type} epoch {epoch}/{num_epochs}  "
                    f"train_MSE={train_loss:.4f}  val_MSE={val_loss:.4f}",
                    flush=True,
                )

        histories[model_type] = hist

    pooling = histories[MODEL_POOLING]
    fourier = histories[MODEL_FOURIER]
    combined = combine_histories(pooling, fourier)
    save_history(paths["pooling"], pooling)
    save_history(paths["fourier"], fourier)
    save_history(paths["combined"], combined)
    if verbose:
        print(f"Wrote {display_path(paths['pooling'])}", flush=True)
        print(f"Wrote {display_path(paths['fourier'])}", flush=True)
        print(f"Wrote {display_path(paths['combined'])}", flush=True)

    return Figure3TrainResult(
        pooling_history=pooling,
        fourier_history=fourier,
        combined_history=combined,
        paths=paths,
        tracked_rows=tuple(tracked_rows),
    )
