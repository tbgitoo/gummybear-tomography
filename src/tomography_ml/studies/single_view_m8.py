"""Single-view M8 study helpers (LR sweep + full train→val/test).

Matches the geometry used in ``notebooks/08_3A_0_learning_rate.ipynb`` and
``notebooks/08_3A_2_train_validation_study.ipynb``: shared ``Encode()`` trunk,
Fourier linear head, flatten MLP, Adam + MSE.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Dataset

from gummybear.datasets.randomization import (
    DEFAULT_SPLIT_FRACTIONS,
    assign_particle_splits,
)
from gummybear.paths import display_path
from tomography_ml.gummybear_data_catalog.IlluminationOnlyDataset import (
    particle_id_from_sequence_id,
)
from tomography_ml.gummybear_data_catalog.catalog import CatalogRow
from tomography_ml.gummybear_data_catalog.task_dataset import (
    DatasetTaskSpec,
    build_task_dataset,
)
from tomography_ml.localization.alternative_localizer import (
    LocalizeSingleView,
    LocalizeSingleViewFlatten,
    LocalizeSingleViewFourier,
)
from tomography_ml.localization.builders import count_parameters, materialize_lazy_modules
from tomography_ml.localization.encoder import Encode
from tomography_ml.studies.study_checkpoints import (
    clone_state_dict,
    load_study_checkpoint,
    save_study_checkpoint,
    should_load_checkpoint,
)
from tomography_ml.training.training_helpers import (
    collect_prediction_errors,
    make_batch_xy_single,
)
from tomography_ml_validation.run_history import (
    aggregate_run_history,
    append_run_history,
    build_history_row,
    effective_n_repeat,
    next_run_id,
    next_seed,
    utc_now_iso,
)

ARCH_ORDER: tuple[str, ...] = ("pooled", "fourier", "flatten")
ARCH_COLORS: dict[str, str] = {
    "pooled": "C0",
    "fourier": "C2",
    "flatten": "C1",
}

# Documented M8 LR selection (used when an LR sweep is skipped).
M8_CANONICAL_LR_BY_ARCH: dict[str, float] = {
    "pooled": 0.001,
    "fourier": 0.03,
    "flatten": 0.0003,
}
# Back-compat alias.
CANONICAL_LR_BY_ARCH = M8_CANONICAL_LR_BY_ARCH



DEFAULT_LR_GRID: tuple[float, ...] = (
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
)

# Default split seeds for the xyz split-sensitivity study (point 4).
DEFAULT_SENSITIVITY_SPLIT_SEEDS: tuple[int, ...] = (60, 61, 62, 63, 64)

ModelFactory = Callable[[str], nn.Module]


def particle_setup_id_for_row(row: CatalogRow) -> str:
    """Particle unit used for split assignment (matches workbook particle_setup_id)."""
    if row.particles:
        return str(row.particles[0].particle_setup_id)
    return particle_id_from_sequence_id(row.sequence_id)


def relabel_catalog_rows_for_split_seed(
    catalog_rows: Sequence[CatalogRow],
    seed: int,
    *,
    split_fractions: Mapping[str, float] | None = None,
) -> list[CatalogRow]:
    """Return catalog rows with train/val/test labels for a new split seed.

    Does **not** rewrite the workbook. Uses
    :func:`~gummybear.datasets.randomization.assign_particle_splits` so all
    sequences sharing a particle id stay in the same partition.
    """
    fractions = dict(DEFAULT_SPLIT_FRACTIONS if split_fractions is None else split_fractions)
    particle_ids = [particle_setup_id_for_row(row) for row in catalog_rows]
    assignment = assign_particle_splits(
        particle_ids,
        seed=int(seed),
        split_fractions=fractions,
    )
    return [
        replace(row, split=str(assignment[particle_setup_id_for_row(row)]))
        for row in catalog_rows
    ]


def set_train_seed(seed: int) -> None:
    """Seed numpy / torch (and CUDA if present) for one training repeat."""
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_m8_single_view_model(
    arch: str,
    *,
    n_outputs: int,
    device: torch.device | str,
) -> nn.Module:
    """Build one M8 single-view localiser (``Encode`` trunk defaults).

    Fourier uses a linear head; flatten uses an MLP (``hidden=128``), matching
    the 08_3A notebooks rather than :func:`builders.make_flatten` defaults
    (which use ``downsample="medium"``).
    """
    arch = str(arch)
    n_outputs = int(n_outputs)
    if arch == "pooled":
        model: nn.Module = LocalizeSingleView(Encode(), n_outputs=n_outputs)
    elif arch == "fourier":
        model = LocalizeSingleViewFourier(
            Encode(),
            n_outputs=n_outputs,
            head_type="linear",
        )
    elif arch == "flatten":
        model = LocalizeSingleViewFlatten(
            Encode(),
            n_outputs=n_outputs,
            hidden=128,
        )
    else:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCH_ORDER}")
    return model.to(device)


# Back-compat alias.
# Back-compat alias.
make_win3a_model = make_m8_single_view_model


def dummy_batch_from_dataset(
    dataset: Dataset,
    *,
    x_field: str,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a ``[1, C, H, W]`` dummy batch from the first dataset sample."""
    x0, _ = dataset[0]
    views0 = np.asarray(x0[x_field])
    if views0.ndim == 4:
        sample = views0[0]
    elif views0.ndim == 3:
        sample = views0
    else:
        sample = views0[None, ...]
    return torch.as_tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)


def probe_m8_parameter_counts(
    *,
    dataset: Dataset,
    x_field: str,
    n_outputs: int,
    device: torch.device | str,
    arch_order: Sequence[str] = ARCH_ORDER,
) -> dict[str, int]:
    """Materialise lazy layers and return trainable parameter counts per arch."""
    dummy = dummy_batch_from_dataset(dataset, x_field=x_field, device=device)
    counts: dict[str, int] = {}
    for arch in arch_order:
        model = make_m8_single_view_model(arch, n_outputs=n_outputs, device=device)
        materialize_lazy_modules(model, dummy)
        counts[str(arch)] = count_parameters(model)
    return counts


# Back-compat alias.
probe_win3a_parameter_counts = probe_m8_parameter_counts


def rmse_metrics_from_l2_errors(
    errs: np.ndarray,
    *,
    y_fields: Sequence[str],
) -> dict[str, float]:
    """History-compatible RMSE dict from per-sample L2 errors.

    When predicting a single axis, that axis column is filled and the others
    stay NaN. For multi-axis targets only ``*_total`` is known from L2 norms
    (per-axis columns remain NaN unless a richer evaluator is used).
    """
    rmse = float(np.sqrt(np.mean(np.asarray(errs, dtype=float) ** 2)))
    metrics = {
        "train_RMSE_total": rmse,
        "train_RMSE_X": float("nan"),
        "train_RMSE_Y": float("nan"),
        "train_RMSE_Z": float("nan"),
    }
    names = [str(f).removeprefix("particle_").upper() for f in y_fields]
    if len(names) == 1 and names[0] in {"X", "Y", "Z"}:
        metrics[f"train_RMSE_{names[0]}"] = rmse
    return metrics


def _mean_mse(
    model: nn.Module,
    loader: DataLoader,
    batch_xy: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for images, targets in loader:
            views, batch_targets = batch_xy(images, targets)
            pred = model(views)
            loss = mse_loss(pred, batch_targets)
            total += float(loss.detach()) * int(views.shape[0])
            count += int(views.shape[0])
    return total / max(count, 1)


def _split_task(task: DatasetTaskSpec, *, name: str, split: str) -> DatasetTaskSpec:
    return replace(
        task,
        name=name,
        row_filter={**dict(task.row_filter), "split": split},
    )


def select_lr_by_arch(
    lr_results: Mapping[str, Mapping[float, Mapping[str, Any]]],
    *,
    arch_order: Sequence[str] = ARCH_ORDER,
) -> dict[str, float]:
    """Pick the LR with lowest finite validation MSE per architecture."""
    selected: dict[str, float] = {}
    for arch in arch_order:
        scored = [
            (float(lr), float(m["val_mse"]))
            for lr, m in lr_results[arch].items()
            if np.isfinite(m["val_mse"])
        ]
        if not scored:
            raise RuntimeError(f"no finite validation MSE for architecture {arch!r}")
        best_lr, _ = min(scored, key=lambda item: item[1])
        selected[str(arch)] = float(best_lr)
    return selected


DEFAULT_LR_STUDY_CHECKPOINT = "m08_learning_rate_study.pt"

FinalStateByArchLr = dict[str, dict[float, dict[str, torch.Tensor]]]


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU clone of ``model.state_dict()`` suitable for ``torch.save``."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _normalize_final_states(
    raw: Mapping[str, Mapping[Any, Mapping[str, torch.Tensor]]],
    *,
    arch_order: Sequence[str],
    lr_results: Mapping[str, Mapping[float, Mapping[str, Any]]],
) -> FinalStateByArchLr:
    """Coerce checkpoint weight tables to ``arch → float(lr) → state_dict``.

    Requires a complete final ``state_dict`` for every successful arch×LR entry
    in ``lr_results`` (failed runs with an ``error`` key may omit weights).
    """
    out: FinalStateByArchLr = {}
    for arch in arch_order:
        arch_key = str(arch)
        by_lr_raw = raw.get(arch_key)
        if by_lr_raw is None:
            raise KeyError(
                f"checkpoint missing final_state_by_arch_lr[{arch_key!r}]"
            )
        normalized = {
            float(lr): {k: v for k, v in state.items()} for lr, state in by_lr_raw.items()
        }
        for lr, metrics in lr_results.get(arch_key, {}).items():
            if "error" in metrics:
                continue
            if float(lr) not in normalized:
                match = next(
                    (k for k in normalized if abs(float(k) - float(lr)) < 1e-15),
                    None,
                )
                if match is None:
                    raise KeyError(
                        f"checkpoint missing final weights for "
                        f"arch={arch_key!r} lr={float(lr):g}"
                    )
        out[arch_key] = normalized
    return out


@dataclass
class LearningRateStudyResult:
    """Outputs of :func:`run_learning_rate_study`."""

    lr_results: dict[str, dict[float, dict[str, Any]]]
    lr_by_arch: dict[str, float]
    train_size: int
    val_size: int
    num_epochs: int
    batch_size: int
    lr_grid: tuple[float, ...]
    y_fields: tuple[str, ...]
    x_field: str
    final_state_by_arch_lr: FinalStateByArchLr
    checkpoint_path: Path | None = None
    skipped_train: bool = False

    def final_state_for(self, arch: str, lr: float | None = None) -> dict[str, torch.Tensor]:
        """Return the final ``state_dict`` for ``arch`` at ``lr`` (default: selected LR)."""
        arch_key = str(arch)
        selected = float(self.lr_by_arch[arch_key] if lr is None else lr)
        by_lr = self.final_state_by_arch_lr[arch_key]
        if selected not in by_lr:
            match = next((k for k in by_lr if abs(float(k) - selected) < 1e-15), None)
            if match is None:
                raise KeyError(
                    f"no final weights for arch={arch_key!r} lr={selected:g}; "
                    f"have {sorted(by_lr)}"
                )
            selected = float(match)
        return by_lr[selected]


def _normalize_lr_results(
    raw: Mapping[str, Mapping[Any, Mapping[str, Any]]],
    *,
    arch_order: Sequence[str],
) -> dict[str, dict[float, dict[str, Any]]]:
    """Coerce checkpoint / in-memory LR tables to ``arch → float(lr) → metrics``."""
    out: dict[str, dict[float, dict[str, Any]]] = {}
    for arch in arch_order:
        arch_key = str(arch)
        by_lr = raw.get(arch_key, {})
        out[arch_key] = {float(lr): dict(metrics) for lr, metrics in by_lr.items()}
    return out


def _learning_rate_study_from_checkpoint(
    blob: Mapping[str, Any],
    *,
    arch_order: Sequence[str],
    checkpoint_path: Path,
) -> LearningRateStudyResult:
    """Rebuild :class:`LearningRateStudyResult` from a saved ``.pt`` payload."""
    if "final_state_by_arch_lr" not in blob:
        raise KeyError(
            f"checkpoint {display_path(checkpoint_path)} is missing "
            "final_state_by_arch_lr (re-run the LR study to write a complete checkpoint)"
        )
    lr_results = _normalize_lr_results(blob["lr_results"], arch_order=arch_order)
    lr_by_arch_raw = blob.get("lr_by_arch") or select_lr_by_arch(
        lr_results, arch_order=arch_order
    )
    lr_by_arch = {str(k): float(v) for k, v in lr_by_arch_raw.items()}
    grid = tuple(float(x) for x in blob.get("lr_grid", ()))
    if not grid:
        grid = tuple(
            sorted({lr for by_lr in lr_results.values() for lr in by_lr.keys()})
        )
    final_states = _normalize_final_states(
        blob["final_state_by_arch_lr"],
        arch_order=arch_order,
        lr_results=lr_results,
    )
    return LearningRateStudyResult(
        lr_results=lr_results,
        lr_by_arch=lr_by_arch,
        train_size=int(blob.get("train_size", 0)),
        val_size=int(blob.get("val_size", 0)),
        num_epochs=int(blob.get("num_epochs", 0)),
        batch_size=int(blob.get("batch_size", 0)),
        lr_grid=grid,
        y_fields=tuple(str(y) for y in blob.get("y_fields", ())),
        x_field=str(blob.get("x_field", "")),
        final_state_by_arch_lr=final_states,
        checkpoint_path=Path(checkpoint_path),
        skipped_train=True,
    )


def run_learning_rate_study(
    *,
    catalog_rows: Sequence[Mapping[str, Any]],
    task: DatasetTaskSpec,
    device: torch.device | str,
    lr_grid: Sequence[float] = DEFAULT_LR_GRID,
    num_epochs: int = 200,
    batch_size: int = 32,
    seed: int = 0,
    arch_order: Sequence[str] = ARCH_ORDER,
    model_factory: ModelFactory | None = None,
    continue_on_failure: bool = True,
    verbose: bool = True,
    results_dir: Path | None = None,
    checkpoint_name: str = DEFAULT_LR_STUDY_CHECKPOINT,
    load_existing: bool = False,
    retrain: bool = False,
) -> LearningRateStudyResult:
    """Sweep learning rates on full train; select by validation MSE (M8 LR study).

    When ``results_dir`` is set, the sweep payload (curves, val MSE, selected
    LRs, and **final** ``state_dict`` per arch×LR) is written to
    ``results_dir / checkpoint_name``. If ``load_existing`` is true and that
    file exists (and ``retrain`` is false), the file is loaded and training is
    skipped.
    """
    if not task.x_fields:
        raise ValueError("task.x_fields must be non-empty")
    x_field = str(task.x_fields[0])
    y_fields = tuple(str(y) for y in task.y_fields)
    n_outputs = len(y_fields)
    factory = model_factory or (
        lambda arch: make_m8_single_view_model(arch, n_outputs=n_outputs, device=device)
    )
    grid = tuple(float(lr) for lr in lr_grid)

    checkpoint_path: Path | None = None
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = results_dir / str(checkpoint_name)
        if (
            bool(load_existing)
            and not bool(retrain)
            and checkpoint_path.is_file()
        ):
            blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if verbose:
                print(
                    f"Loaded LR study from {display_path(checkpoint_path)}",
                    flush=True,
                )
            return _learning_rate_study_from_checkpoint(
                blob,
                arch_order=arch_order,
                checkpoint_path=checkpoint_path,
            )

    train_ds = build_task_dataset(catalog_rows, task)
    val_ds = build_task_dataset(
        catalog_rows,
        _split_task(task, name=f"{task.name}_val_lr", split="validation"),
    )
    if len(train_ds) < 1 or len(val_ds) < 1:
        raise ValueError(
            f"need non-empty train/val for LR study; got train={len(train_ds)} val={len(val_ds)}"
        )
    if verbose:
        print(f"LR study: train={len(train_ds)}  validation={len(val_ds)}", flush=True)

    batch_xy = make_batch_xy_single(
        x_field=x_field,
        y_fields=y_fields,
        device=device,
    )
    lr_results: dict[str, dict[float, dict[str, Any]]] = {a: {} for a in arch_order}
    final_state_by_arch_lr: FinalStateByArchLr = {str(a): {} for a in arch_order}

    for arch in arch_order:
        if verbose:
            print(f"\n=== {arch} ===", flush=True)
        for lr in grid:
            try:
                model = factory(str(arch))
                train_loader = DataLoader(
                    train_ds,
                    batch_size=min(int(batch_size), len(train_ds)),
                    shuffle=True,
                    generator=torch.Generator().manual_seed(int(seed)),
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=min(int(batch_size), len(val_ds)),
                    shuffle=False,
                )
                with torch.no_grad():
                    images0, targets0 = next(iter(train_loader))
                    v0, _ = batch_xy(images0, targets0)
                    _ = model(v0)
                opt = torch.optim.Adam(model.parameters(), lr=float(lr))
                train_curve: list[float] = []
                for _epoch in range(int(num_epochs)):
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
                    train_curve.append(total / max(count, 1))
                metrics = {
                    "train_curve": train_curve,
                    "train_mse": _mean_mse(model, train_loader, batch_xy),
                    "val_mse": _mean_mse(model, val_loader, batch_xy),
                    "n_params": count_parameters(model),
                }
                final_state_by_arch_lr[str(arch)][float(lr)] = _clone_state_dict(model)
            except Exception as exc:  # noqa: BLE001 — keep sweep going on blow-ups
                if not continue_on_failure:
                    raise
                metrics = {
                    "train_curve": [float("nan")] * int(num_epochs),
                    "train_mse": float("nan"),
                    "val_mse": float("nan"),
                    "n_params": -1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if verbose:
                    print(
                        f"  lr={lr:g}  FAILED: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
            lr_results[str(arch)][float(lr)] = metrics
            if verbose and "error" not in metrics:
                print(
                    f"  lr={lr:g}  train_MSE={metrics['train_mse']:.6g}  "
                    f"val_MSE={metrics['val_mse']:.6g}",
                    flush=True,
                )

    lr_by_arch = select_lr_by_arch(lr_results, arch_order=arch_order)
    if checkpoint_path is not None:
        payload = {
            "lr_results": lr_results,
            "lr_by_arch": lr_by_arch,
            "final_state_by_arch_lr": final_state_by_arch_lr,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "lr_grid": list(grid),
            "y_fields": list(y_fields),
            "x_field": x_field,
            "seed": int(seed),
            "arch_order": list(arch_order),
        }
        torch.save(payload, checkpoint_path)
        if verbose:
            print(f"Wrote LR study checkpoint {display_path(checkpoint_path)}", flush=True)

    return LearningRateStudyResult(
        lr_results=lr_results,
        lr_by_arch=lr_by_arch,
        train_size=len(train_ds),
        val_size=len(val_ds),
        num_epochs=int(num_epochs),
        batch_size=int(batch_size),
        lr_grid=grid,
        y_fields=y_fields,
        x_field=x_field,
        final_state_by_arch_lr=final_state_by_arch_lr,
        checkpoint_path=checkpoint_path,
        skipped_train=False,
    )


@dataclass
class TrainValTestStudyResult:
    """Outputs of :func:`run_train_val_test_study`."""

    full_results: dict[str, dict[str, Any]]
    session_run_ids: list[int]
    session_summary_df: pd.DataFrame
    comparison_df: pd.DataFrame
    history_path: Path
    session_summary_path: Path
    comparison_path: Path
    train_size: int
    val_size: int
    test_size: int
    n_rep: int
    lr_by_arch: dict[str, float]
    y_fields: tuple[str, ...]
    x_field: str
    final_state_by_arch: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    skipped_train: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _train_val_test_from_checkpoint(
    blob: Mapping[str, Any],
    *,
    results_dir: Path,
    csv_run_history: str,
    csv_session_summary: str,
    csv_comparison: str,
    checkpoint_path: Path,
) -> TrainValTestStudyResult:
    """Rebuild :class:`TrainValTestStudyResult` from a checkpoint payload."""
    if "final_state_by_arch" not in blob:
        raise KeyError(
            f"checkpoint {display_path(checkpoint_path)} missing final_state_by_arch"
        )
    if "full_results" not in blob:
        raise KeyError(
            f"checkpoint {display_path(checkpoint_path)} missing full_results"
        )
    session_summary_df = pd.DataFrame(blob["session_summary"])
    comparison_df = pd.DataFrame(blob["comparison"])
    history_path = results_dir / csv_run_history
    session_summary_path = results_dir / csv_session_summary
    comparison_path = results_dir / csv_comparison
    results_dir.mkdir(parents=True, exist_ok=True)
    session_summary_df.to_csv(session_summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    if "history_rows" in blob:
        pd.DataFrame(blob["history_rows"]).to_csv(history_path, index=False)
    elif not history_path.is_file():
        # Minimal history stub so downstream loaders still find a path.
        session_summary_df.to_csv(history_path, index=False)

    final_state_by_arch = {
        str(arch): {k: v for k, v in state.items()}
        for arch, state in dict(blob["final_state_by_arch"]).items()
    }
    return TrainValTestStudyResult(
        full_results=dict(blob["full_results"]),
        session_run_ids=[int(x) for x in blob.get("session_run_ids", [])],
        session_summary_df=session_summary_df,
        comparison_df=comparison_df,
        history_path=history_path,
        session_summary_path=session_summary_path,
        comparison_path=comparison_path,
        train_size=int(blob.get("train_size", 0)),
        val_size=int(blob.get("val_size", 0)),
        test_size=int(blob.get("test_size", 0)),
        n_rep=int(blob.get("n_rep", 0)),
        lr_by_arch={str(k): float(v) for k, v in dict(blob.get("lr_by_arch", {})).items()},
        y_fields=tuple(str(y) for y in blob.get("y_fields", ())),
        x_field=str(blob.get("x_field", "")),
        final_state_by_arch=final_state_by_arch,
        checkpoint_path=Path(checkpoint_path),
        skipped_train=True,
        extra=dict(blob.get("extra", {})),
    )


def run_train_val_test_study(
    *,
    catalog_rows: Sequence[Mapping[str, Any]],
    task: DatasetTaskSpec,
    device: torch.device | str,
    results_dir: Path,
    lr_by_arch: Mapping[str, float],
    notebook_id: str,
    experiment_id: str,
    variant_prefix: str,
    num_epochs: int = 200,
    batch_size: int = 32,
    n_repeat_training: int = 3,
    base_seed: int = 0,
    arch_order: Sequence[str] = ARCH_ORDER,
    model_factory: ModelFactory | None = None,
    csv_run_history: str = "run_history.csv",
    csv_session_summary: str = "session_summary.csv",
    csv_comparison: str = "comparison_last_run.csv",
    architecture_note_prefix: str = "M8 single-view",
    history_extra: Mapping[str, Any] | None = None,
    verbose: bool = True,
    checkpoint_name: str | None = None,
    load_existing: bool = False,
    retrain: bool = False,
) -> TrainValTestStudyResult:
    """Train each architecture on full train; evaluate val/test (M8 train→val/test).

    When ``checkpoint_name`` is set, the final per-arch weights plus plot/table
    payloads are stored under ``results_dir / checkpoint_name``. Loading skips
    training when ``load_existing`` is true and the file exists.
    """
    if not task.x_fields:
        raise ValueError("task.x_fields must be non-empty")
    x_field = str(task.x_fields[0])
    y_fields = tuple(str(y) for y in task.y_fields)
    n_outputs = len(y_fields)
    factory = model_factory or (
        lambda arch: make_m8_single_view_model(arch, n_outputs=n_outputs, device=device)
    )
    extra_cols = dict(history_extra or {})

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        results_dir / str(checkpoint_name) if checkpoint_name else None
    )
    if should_load_checkpoint(
        checkpoint_path, load_existing=load_existing, retrain=retrain
    ):
        assert checkpoint_path is not None
        blob = load_study_checkpoint(checkpoint_path)
        if verbose:
            print(
                f"Loaded train/val/test study from {display_path(checkpoint_path)}",
                flush=True,
            )
        return _train_val_test_from_checkpoint(
            blob,
            results_dir=results_dir,
            csv_run_history=csv_run_history,
            csv_session_summary=csv_session_summary,
            csv_comparison=csv_comparison,
            checkpoint_path=checkpoint_path,
        )

    train_ds = build_task_dataset(catalog_rows, task)
    val_ds = build_task_dataset(
        catalog_rows,
        _split_task(task, name=f"{task.name}_val", split="validation"),
    )
    test_ds = build_task_dataset(
        catalog_rows,
        _split_task(task, name=f"{task.name}_test", split="test"),
    )
    if len(train_ds) < 1 or len(val_ds) < 1 or len(test_ds) < 1:
        raise ValueError(
            "need non-empty train, validation, and test; "
            f"got train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
        )

    batch_xy = make_batch_xy_single(
        x_field=x_field,
        y_fields=y_fields,
        device=device,
    )
    n_rep = effective_n_repeat(int(n_repeat_training))
    history_path = results_dir / csv_run_history
    session_base_seed = next_seed(history_path, default=int(base_seed))
    if verbose:
        print(
            f"train={len(train_ds)}  validation={len(val_ds)}  test={len(test_ds)}  "
            f"n_rep={n_rep}  session_base_seed={session_base_seed}  "
            f"y={list(y_fields)}",
            flush=True,
        )

    session_run_ids: list[int] = []
    session_history_rows: list[dict[str, Any]] = []
    full_results: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    final_state_by_arch: dict[str, dict[str, torch.Tensor]] = {}

    for rep in range(n_rep):
        seed = int(session_base_seed) + int(rep)
        set_train_seed(seed)
        run_id = (
            max(session_run_ids) + 1
            if session_run_ids
            else next_run_id(history_path)
        )
        session_run_ids.append(run_id)
        ts = utc_now_iso()
        rep_results: dict[str, dict[str, Any]] = {}
        if verbose:
            print(
                f"\n######## Train+eval repeat {rep + 1}/{n_rep}  "
                f"run_id={run_id}  seed={seed}",
                flush=True,
            )

        for arch in arch_order:
            lr = float(lr_by_arch[str(arch)])
            model = factory(str(arch))
            train_loader = DataLoader(
                train_ds,
                batch_size=min(int(batch_size), len(train_ds)),
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=min(int(batch_size), len(val_ds)),
                shuffle=False,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=min(int(batch_size), len(test_ds)),
                shuffle=False,
            )
            with torch.no_grad():
                images0, targets0 = next(iter(train_loader))
                v0, _ = batch_xy(images0, targets0)
                _ = model(v0)
            n_params = count_parameters(model)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            train_loss_curve: list[float] = []
            for _epoch in range(int(num_epochs)):
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
                train_loss_curve.append(total / max(count, 1))

            train_errs, train_mse = collect_prediction_errors(
                model, train_loader, batch_xy
            )
            val_errs, val_mse = collect_prediction_errors(model, val_loader, batch_xy)
            test_errs, test_mse = collect_prediction_errors(
                model, test_loader, batch_xy
            )
            train_rmse = float(np.sqrt(np.mean(train_errs**2)))
            val_rmse = float(np.sqrt(np.mean(val_errs**2)))
            test_rmse = float(np.sqrt(np.mean(test_errs**2)))
            variant_id = f"{variant_prefix}_{arch}"
            rep_results[str(arch)] = {
                "n_params": n_params,
                "lr": lr,
                "seed": seed,
                "train_loss_curve": train_loss_curve,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "train_errs": train_errs,
                "val_errs": val_errs,
                "test_errs": test_errs,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "test_rmse": test_rmse,
                "variant_id": variant_id,
            }
            if rep == n_rep - 1:
                final_state_by_arch[str(arch)] = clone_state_dict(model)
            if verbose:
                print(
                    f"{arch:8s}  params={n_params:,}  lr={lr:g}  seed={seed}  "
                    f"train_MSE={train_mse:.4f}  val_MSE={val_mse:.4f}  "
                    f"test_MSE={test_mse:.4f}  "
                    f"train_RMSE={train_rmse:.4f}  val_RMSE={val_rmse:.4f}  "
                    f"test_RMSE={test_rmse:.4f}",
                    flush=True,
                )
            session_history_rows.append(
                build_history_row(
                    run_id=run_id,
                    repeat_index=rep,
                    timestamp_utc=ts,
                    notebook_id=notebook_id,
                    experiment_id=experiment_id,
                    variant_id=variant_id,
                    fusion_pattern="single_view",
                    backbone_kind=str(arch),
                    metrics_by_split={
                        "train": rmse_metrics_from_l2_errors(
                            train_errs, y_fields=y_fields
                        ),
                        "validation": rmse_metrics_from_l2_errors(
                            val_errs, y_fields=y_fields
                        ),
                        "test": rmse_metrics_from_l2_errors(
                            test_errs, y_fields=y_fields
                        ),
                    },
                    architecture_note=f"{architecture_note_prefix} {arch} spatial readout",
                    learned_parameter_count=n_params,
                    n_views=1,
                    lr_stage_a=lr,
                    epochs_max=int(num_epochs),
                    epochs_ran_stage_a=int(num_epochs),
                    batch_size=int(batch_size),
                    n_repeat_training_requested=int(n_repeat_training),
                    n_repeat_training_effective=int(n_rep),
                    seed=seed,
                    freeze_encoder=False,
                    end_to_end=True,
                    extra={
                        "train_MSE": train_mse,
                        "validation_MSE": val_mse,
                        "test_MSE": test_mse,
                        "y_fields": ",".join(y_fields),
                        **extra_cols,
                    },
                )
            )
            if rep == n_rep - 1:
                comparison_rows.append(
                    {
                        "variant_id": variant_id,
                        "backbone_kind": arch,
                        "seed": seed,
                        "run_id": run_id,
                        "n_params": n_params,
                        "lr": lr,
                        "train_MSE": train_mse,
                        "validation_MSE": val_mse,
                        "test_MSE": test_mse,
                        "train_RMSE": train_rmse,
                        "validation_RMSE": val_rmse,
                        "test_RMSE": test_rmse,
                        "y_fields": ",".join(y_fields),
                        **extra_cols,
                    }
                )

        full_results = rep_results

    append_run_history(history_path, session_history_rows)
    session_summary_df = aggregate_run_history(pd.DataFrame(session_history_rows))
    session_summary_path = results_dir / csv_session_summary
    session_summary_df.to_csv(session_summary_path, index=False)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = results_dir / csv_comparison
    comparison_df.to_csv(comparison_path, index=False)

    if checkpoint_path is not None:
        save_study_checkpoint(
            checkpoint_path,
            {
                "full_results": full_results,
                "session_summary": session_summary_df.to_dict(orient="list"),
                "comparison": comparison_df.to_dict(orient="list"),
                "history_rows": session_history_rows,
                "session_run_ids": session_run_ids,
                "final_state_by_arch": final_state_by_arch,
                "lr_by_arch": dict(lr_by_arch),
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "test_size": len(test_ds),
                "n_rep": int(n_rep),
                "y_fields": list(y_fields),
                "x_field": x_field,
                "extra": dict(extra_cols),
            },
        )
        if verbose:
            print(
                f"Wrote train/val/test checkpoint {display_path(checkpoint_path)}",
                flush=True,
            )

    return TrainValTestStudyResult(
        full_results=full_results,
        session_run_ids=session_run_ids,
        session_summary_df=session_summary_df,
        comparison_df=comparison_df,
        history_path=history_path,
        session_summary_path=session_summary_path,
        comparison_path=comparison_path,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
        n_rep=int(n_rep),
        lr_by_arch=dict(lr_by_arch),
        y_fields=y_fields,
        x_field=x_field,
        final_state_by_arch=final_state_by_arch,
        checkpoint_path=checkpoint_path,
        skipped_train=False,
    )


@dataclass
class SplitSensitivityStudyResult:
    """Outputs of :func:`run_split_sensitivity_study`."""

    split_seeds: tuple[int, ...]
    per_seed_studies: dict[int, TrainValTestStudyResult]
    per_seed_metrics: pd.DataFrame
    summary_df: pd.DataFrame
    summary_path: Path
    per_seed_path: Path
    results_dir: Path
    lr_by_arch: dict[str, float]
    y_fields: tuple[str, ...]
    x_field: str
    final_state_by_seed_arch: dict[int, dict[str, dict[str, torch.Tensor]]] = field(
        default_factory=dict
    )
    checkpoint_path: Path | None = None
    skipped_train: bool = False


def _summary_df_from_split_metrics(
    per_seed_metrics: pd.DataFrame,
    *,
    variant_prefix: str,
    arch_order: Sequence[str] = ARCH_ORDER,
) -> pd.DataFrame:
    """Build a ``plot_rmse_summary_bars``-compatible summary across split seeds."""
    rows: list[dict[str, Any]] = []
    for arch in arch_order:
        sub = per_seed_metrics.loc[per_seed_metrics["backbone_kind"] == arch]
        n_runs = int(len(sub))
        rows.append(
            {
                "variant_id": f"{variant_prefix}_{arch}",
                "backbone_kind": arch,
                "fusion_pattern": "single_view",
                "n_runs": n_runs,
                "run_ids": ",".join(str(int(s)) for s in sub["split_seed"].tolist()),
                "RMSE_validation_mean": float(sub["validation_RMSE"].mean()),
                "RMSE_validation_std": float(sub["validation_RMSE"].std(ddof=1))
                if n_runs >= 2
                else 0.0,
                "RMSE_test_mean": float(sub["test_RMSE"].mean()),
                "RMSE_test_std": float(sub["test_RMSE"].std(ddof=1))
                if n_runs >= 2
                else 0.0,
                "RMSE_train_mean": float(sub["train_RMSE"].mean()),
                "RMSE_train_std": float(sub["train_RMSE"].std(ddof=1))
                if n_runs >= 2
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_split_sensitivity_study(
    *,
    catalog_rows: Sequence[CatalogRow],
    task: DatasetTaskSpec,
    device: torch.device | str,
    results_dir: Path,
    lr_by_arch: Mapping[str, float],
    split_seeds: Sequence[int] = DEFAULT_SENSITIVITY_SPLIT_SEEDS,
    split_fractions: Mapping[str, float] | None = None,
    notebook_id: str = "08_3A_2",
    experiment_id: str = "m08_3a2_xyz_split_sensitivity",
    variant_prefix: str = "m08_3a2xyz_sens",
    num_epochs: int = 200,
    batch_size: int = 32,
    training_seed: int = 0,
    arch_order: Sequence[str] = ARCH_ORDER,
    model_factory: ModelFactory | None = None,
    architecture_note_prefix: str = "M8 xyz split-sensitivity",
    verbose: bool = True,
    checkpoint_name: str | None = None,
    load_existing: bool = False,
    retrain: bool = False,
) -> SplitSensitivityStudyResult:
    """Repeat xyz (or any) train→val/test once per split seed (one training run each).

    Relabels catalog rows in memory for each seed; the live workbook is unchanged.
    Training RNG seed is fixed (``training_seed``) so variability is from the
    particle-level split assignment.

    When ``checkpoint_name`` is set, metrics tables plus final per-seed/arch
    weights are stored under ``results_dir / checkpoint_name``.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in split_seeds)
    if not seeds:
        raise ValueError("split_seeds must be non-empty")

    checkpoint_path = (
        results_dir / str(checkpoint_name) if checkpoint_name else None
    )
    y_fields = tuple(str(y) for y in task.y_fields)
    x_field = str(task.x_fields[0])
    per_seed_path = results_dir / "sensitivity_per_seed.csv"
    summary_path = results_dir / "sensitivity_summary.csv"

    if should_load_checkpoint(
        checkpoint_path, load_existing=load_existing, retrain=retrain
    ):
        assert checkpoint_path is not None
        blob = load_study_checkpoint(checkpoint_path)
        if "final_state_by_seed_arch" not in blob:
            raise KeyError(
                f"checkpoint {display_path(checkpoint_path)} missing "
                "final_state_by_seed_arch"
            )
        per_seed_metrics = pd.DataFrame(blob["per_seed_metrics"])
        summary_df = pd.DataFrame(blob["summary"])
        per_seed_metrics.to_csv(per_seed_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        final_state_by_seed_arch = {
            int(seed): {
                str(arch): {k: v for k, v in state.items()}
                for arch, state in by_arch.items()
            }
            for seed, by_arch in dict(blob["final_state_by_seed_arch"]).items()
        }
        if verbose:
            print(
                f"Loaded split-sensitivity study from {display_path(checkpoint_path)}",
                flush=True,
            )
        return SplitSensitivityStudyResult(
            split_seeds=tuple(int(s) for s in blob.get("split_seeds", seeds)),
            per_seed_studies={},
            per_seed_metrics=per_seed_metrics,
            summary_df=summary_df,
            summary_path=summary_path,
            per_seed_path=per_seed_path,
            results_dir=results_dir,
            lr_by_arch={
                str(k): float(v) for k, v in dict(blob.get("lr_by_arch", {})).items()
            },
            y_fields=tuple(str(y) for y in blob.get("y_fields", y_fields)),
            x_field=str(blob.get("x_field", x_field)),
            final_state_by_seed_arch=final_state_by_seed_arch,
            checkpoint_path=checkpoint_path,
            skipped_train=True,
        )

    per_seed_studies: dict[int, TrainValTestStudyResult] = {}
    metric_rows: list[dict[str, Any]] = []
    final_state_by_seed_arch: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

    for split_seed in seeds:
        if verbose:
            print(
                f"\n======== Split sensitivity  split_seed={split_seed}  "
                f"(training_seed={training_seed}, n_repeat=1) ========",
                flush=True,
            )
        relabeled = relabel_catalog_rows_for_split_seed(
            catalog_rows,
            split_seed,
            split_fractions=split_fractions,
        )
        seed_dir = results_dir / f"split_seed_{split_seed}"
        study = run_train_val_test_study(
            catalog_rows=relabeled,
            task=task,
            device=device,
            results_dir=seed_dir,
            lr_by_arch=lr_by_arch,
            notebook_id=notebook_id,
            experiment_id=experiment_id,
            variant_prefix=variant_prefix,
            num_epochs=num_epochs,
            batch_size=batch_size,
            n_repeat_training=1,
            base_seed=int(training_seed),
            arch_order=arch_order,
            model_factory=model_factory,
            csv_run_history="run_history.csv",
            csv_session_summary="session_summary.csv",
            csv_comparison="comparison_last_run.csv",
            architecture_note_prefix=architecture_note_prefix,
            history_extra={"split_seed": int(split_seed)},
            verbose=verbose,
        )
        per_seed_studies[int(split_seed)] = study
        final_state_by_seed_arch[int(split_seed)] = dict(study.final_state_by_arch)
        for arch in arch_order:
            row = study.full_results[str(arch)]
            metric_rows.append(
                {
                    "split_seed": int(split_seed),
                    "backbone_kind": str(arch),
                    "variant_id": f"{variant_prefix}_{arch}",
                    "train_size": study.train_size,
                    "val_size": study.val_size,
                    "test_size": study.test_size,
                    "train_RMSE": float(row["train_rmse"]),
                    "validation_RMSE": float(row["val_rmse"]),
                    "test_RMSE": float(row["test_rmse"]),
                    "train_MSE": float(row["train_mse"]),
                    "validation_MSE": float(row["val_mse"]),
                    "test_MSE": float(row["test_mse"]),
                    "lr": float(row["lr"]),
                    "n_params": int(row["n_params"]),
                    "training_seed": int(training_seed),
                }
            )

    per_seed_metrics = pd.DataFrame(metric_rows)
    summary_df = _summary_df_from_split_metrics(
        per_seed_metrics,
        variant_prefix=variant_prefix,
        arch_order=arch_order,
    )
    per_seed_metrics.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if checkpoint_path is not None:
        save_study_checkpoint(
            checkpoint_path,
            {
                "split_seeds": list(seeds),
                "per_seed_metrics": per_seed_metrics.to_dict(orient="list"),
                "summary": summary_df.to_dict(orient="list"),
                "final_state_by_seed_arch": final_state_by_seed_arch,
                "lr_by_arch": dict(lr_by_arch),
                "y_fields": list(y_fields),
                "x_field": x_field,
                "training_seed": int(training_seed),
            },
        )
        if verbose:
            print(
                f"Wrote split-sensitivity checkpoint {display_path(checkpoint_path)}",
                flush=True,
            )

    return SplitSensitivityStudyResult(
        split_seeds=seeds,
        per_seed_studies=per_seed_studies,
        per_seed_metrics=per_seed_metrics,
        summary_df=summary_df,
        summary_path=summary_path,
        per_seed_path=per_seed_path,
        results_dir=results_dir,
        lr_by_arch=dict(lr_by_arch),
        y_fields=y_fields,
        x_field=x_field,
        final_state_by_seed_arch=final_state_by_seed_arch,
        checkpoint_path=checkpoint_path,
        skipped_train=False,
    )
