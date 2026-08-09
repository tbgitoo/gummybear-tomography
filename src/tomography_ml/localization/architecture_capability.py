"""Architecture capability helpers for single-view localisation studies.

Three capability stages support fair architecture comparison:

1. **Overfitability** — train on tiny fixed train subsets to test whether a
   model can memorise coordinates (capacity vs collapse).
2. **Aggregation** — summarise overfit runs across architectures and subset sizes.
3. **Full-split training** — train on the complete train split with optional
   validation early stopping.

Shared train subsets ensure differences reflect model capacity, not sample luck.
Train/validation/test splits are always by ``sequence_id``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from tomography_ml.localization.builders import (
    SingleViewArchConfig,
    build_from_config,
    count_parameters,
    describe_feature_geometry,
    materialize_lazy_modules,
)


@dataclass(frozen=True)
class SharedSubset:
    """Fixed train-split subset shared across architecture variants.

    The same ``indices`` / ``sequence_ids`` must be reused for every architecture
    in an overfitability sweep so differences reflect capacity, not sample
    luck. Build subsets from the **train split only** — never validation or test
    rows (splits are by ``sequence_id``).

    Attributes:
        subset_id: Stable label ``n{n}_rep{rep}_seed{seed}``.
        n: Number of train sequences in this subset.
        rep: Repeat index for the same ``n`` (multiple random draws).
        indices: Row indices into the train-pool ordering used at build time.
        sequence_ids: ``sequence_id`` strings for the selected rows (audit trail).
        seed: RNG seed used to draw ``indices``.
        input_representation: Logged ``x_field`` / representation name.
        normalisation: Logged ``image_normalize`` mode.
    """

    subset_id: str
    n: int
    rep: int
    indices: tuple[int, ...]
    sequence_ids: tuple[str, ...]
    seed: int
    input_representation: str
    normalisation: str

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for CSV table / JSON file logging; ``indices`` remain a tuple."""
        return asdict(self)


def build_shared_subsets(
    *,
    n_pool: int,
    sequence_ids: Sequence[str],
    ns: Sequence[int] = (1, 2, 3, 5, 10),
    n_reps: int = 3,
    base_seed: int = 0,
    input_representation: str = "anomaly_ref",
    normalisation: str = "none",
) -> tuple[SharedSubset, ...]:
    """Create fixed shared train subsets for overfitability testing (stage 1).

    For each subset size in ``ns`` and each repeat, draws ``n`` distinct indices
    without replacement from ``[0, n_pool)``. The same ``(n, rep)`` indices are
    reused for every architecture variant in a sweep.

    Preconditions:
        - ``sequence_ids`` must list **train-split** ``sequence_id`` values only.
        - ``len(sequence_ids) == n_pool``.
        - Never call with validation or test rows.

    Args:
        n_pool: Train-pool size (number of train sequences).
        sequence_ids: ``sequence_id`` for each train-pool row, index-aligned.
        ns: Subset sizes to sweep (each must be ``1 ≤ n ≤ n_pool``).
        n_reps: Independent random draws per ``n``.
        base_seed: Master seed; per-subset seeds are derived deterministically.
        input_representation: Logged representation / ``x_field`` label.
        normalisation: Logged normalisation mode.

    Returns:
        Tuple of :class:`SharedSubset`, ordered by ``n`` then ``rep``.
    """
    if n_pool < 1:
        raise ValueError(f"n_pool must be >= 1; got {n_pool}")
    if len(sequence_ids) != n_pool:
        raise ValueError(
            "sequence_ids length must equal n_pool; "
            f"got {len(sequence_ids)} vs {n_pool}"
        )
    rng = np.random.default_rng(base_seed)
    subsets: list[SharedSubset] = []
    for n in ns:
        n_i = int(n)
        if n_i < 1:
            raise ValueError(f"subset sizes must be >= 1; got {n}")
        if n_i > n_pool:
            raise ValueError(
                f"subset size n={n_i} exceeds train pool size {n_pool}"
            )
        for rep in range(int(n_reps)):
            seed = int(base_seed + 1000 * n_i + rep)
            # Draw with a dedicated RNG stream so subset defs stay stable
            # even if callers change iteration order over architectures.
            draw_rng = np.random.default_rng(seed)
            indices = tuple(
                int(i)
                for i in draw_rng.choice(n_pool, size=n_i, replace=False)
            )
            # Consume from master rng as well to keep a logged stream id.
            _ = rng.integers(0, 2**31 - 1)
            seqs = tuple(str(sequence_ids[i]) for i in indices)
            subsets.append(
                SharedSubset(
                    subset_id=f"n{n_i}_rep{rep}_seed{seed}",
                    n=n_i,
                    rep=int(rep),
                    indices=indices,
                    sequence_ids=seqs,
                    seed=seed,
                    input_representation=str(input_representation),
                    normalisation=str(normalisation),
                )
            )
    return tuple(subsets)


def _axis_names(y_fields: Sequence[str]) -> list[str]:
    names: list[str] = []
    for field in y_fields:
        if field.startswith("particle_"):
            names.append(field.replace("particle_", "", 1).upper())
        else:
            names.append(str(field))
    return names


def per_axis_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_fields: Sequence[str],
) -> dict[str, float]:
    """Compute total and per-axis root-mean-square error (RMSE) for localisation targets.

    Axis names are inferred from ``y_fields`` (``particle_x`` → ``X``). Missing
    X/Y/Z columns are left as ``NaN`` in the canonical keys.

    Args:
        pred: Predictions ``[N, n_out]``.
        target: Ground truth, same shape as ``pred``.
        y_fields: Target field names in column order.

    Returns:
        Dict with ``train_RMSE_total``, ``train_RMSE_X/Y/Z`` (when present),
        and ``train_RMSE_axis{i}_{name}`` for every column.

    Raises:
        ValueError: If ``pred`` and ``target`` shapes differ.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    diff = pred.detach().float() - target.detach().float()
    total = float(torch.sqrt((diff**2).mean()).item())
    axis_names = _axis_names(y_fields)
    out: dict[str, float] = {
        "train_RMSE_total": total,
        "train_RMSE_X": float("nan"),
        "train_RMSE_Y": float("nan"),
        "train_RMSE_Z": float("nan"),
    }
    for i, name in enumerate(axis_names):
        rmse_i = float(torch.sqrt((diff[:, i] ** 2).mean()).item())
        key = f"train_RMSE_{name}" if name in {"X", "Y", "Z"} else None
        if key is not None:
            out[key] = rmse_i
        out[f"train_RMSE_axis{i}_{name}"] = rmse_i
    return out


def constant_mean_target_baseline_rmse(targets: torch.Tensor) -> float:
    """Root-mean-square error (RMSE) of predicting the per-coordinate mean of ``targets``.

    Sanity baseline: a model that outputs the train-subset mean for
    each axis. Used with :func:`overfit_success` to detect genuine overfit
    vs. collapse.

    Args:
        targets: Ground-truth tensor ``[N, n_out]``.

    Returns:
        Scalar root-mean-square error (RMSE).
    """
    t = targets.detach().float()
    mean = t.mean(dim=0, keepdim=True).expand_as(t)
    return float(torch.sqrt(((t - mean) ** 2).mean()).item())


def collapse_to_mean_target(
    pred: torch.Tensor,
    targets: torch.Tensor,
    *,
    relative_tol: float = 0.05,
) -> bool:
    """Detect midpoint collapse (predictions ≈ coordinate-wise mean).

    Args:
        pred: Model outputs ``[N, n_out]``.
        targets: Ground truth, same shape.
        relative_tol: Collapse if prediction spread ≤ this fraction of target
            spread about the mean.

    Returns:
        ``True`` when predictions stay near the mean target (failed overfit).
    """
    t = targets.detach().float()
    p = pred.detach().float()
    mean = t.mean(dim=0, keepdim=True)
    pred_spread = float(torch.sqrt(((p - mean) ** 2).mean()).item())
    target_spread = float(torch.sqrt(((t - mean) ** 2).mean()).item())
    if target_spread < 1e-8:
        return pred_spread < 1e-6
    return pred_spread <= relative_tol * target_spread


def pairwise_distance_stats(
    coords: torch.Tensor,
) -> dict[str, float]:
    """Summarise pairwise Euclidean (L2) distances between target coordinates.

    Diagnostic for tiny train subsets: distinguishes spread in ground truth
    from spread in predictions.

    Args:
        coords: Coordinate tensor ``[N, n_dims]`` with ``N ≥ 1``.

    Returns:
        Dict with ``pairwise_mean``, ``pairwise_min``, ``pairwise_max``;
        all ``NaN`` when ``N < 2``.
    """
    x = coords.detach().float()
    if x.shape[0] < 2:
        return {
            "pairwise_mean": float("nan"),
            "pairwise_min": float("nan"),
            "pairwise_max": float("nan"),
        }
    # [n, n]
    d = torch.cdist(x, x, p=2)
    iu = torch.triu_indices(d.shape[0], d.shape[1], offset=1)
    vals = d[iu[0], iu[1]]
    return {
        "pairwise_mean": float(vals.mean().item()),
        "pairwise_min": float(vals.min().item()),
        "pairwise_max": float(vals.max().item()),
    }


def overfit_success(
    *,
    train_rmse_total: float,
    baseline_rmse: float,
    rmse_threshold: float | None = None,
    fraction_of_baseline: float = 0.25,
) -> bool:
    """Overfit success heuristic for architecture comparison (document threshold).

    Default rule: success when ``train_rmse_total < fraction_of_baseline *
    baseline_rmse`` (typically 25% of the mean-target baseline). Override with
    an absolute ``rmse_threshold`` when a fixed cutoff is preferred.

    Args:
        train_rmse_total: Final train root-mean-square error (RMSE) from
            :func:`train_subset_run`.
        baseline_rmse: From :func:`constant_mean_target_baseline_rmse`.
        rmse_threshold: Optional absolute RMSE ceiling.
        fraction_of_baseline: Relative cutoff when ``rmse_threshold`` is ``None``.

    Returns:
        ``True`` when the architecture cleared the chosen criterion.
    """
    if rmse_threshold is not None:
        return bool(train_rmse_total < float(rmse_threshold))
    return bool(train_rmse_total < float(fraction_of_baseline) * float(baseline_rmse))


BatchXY = Callable[[Any, Any], tuple[torch.Tensor, torch.Tensor]]


def train_subset_run(
    *,
    model: nn.Module,
    loader: DataLoader,
    batch_xy: BatchXY,
    device: torch.device | str,
    y_fields: Sequence[str],
    num_epochs: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    """Train on one tiny train subset and return overfit diagnostics.

    Fits with Adam on **train rows only** (no validation). Intended for
    stage-1 overfitability testing — not for full-split or test evaluation.

    Args:
        model: Fresh localiser for this architecture × subset cell.
        loader: DataLoader yielding only the subset's train sequences.
        batch_xy: Maps ``(images, targets)`` dict batches to ``(views, y)``.
        device: Torch device (unused directly — tensors should already match).
        y_fields: Target column order for root-mean-square error (RMSE)
            reporting.
        num_epochs: Full-epoch count (no early stopping).
        lr: Adam learning rate (LR).
        seed: ``torch.manual_seed`` value (subset ``seed`` for reproducibility).

    Returns:
        Dict with ``final_train_loss``, per-axis ``train_RMSE_*``,
        ``constant_mean_target_baseline``, ``collapse_to_mean_target``,
        ``overfit_success``, pairwise distance stats, and coordinate lists.
    """
    torch.manual_seed(int(seed))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    mean_loss = float("nan")
    for _epoch in range(int(num_epochs)):
        total = 0.0
        count = 0
        for images, targets in loader:
            views, batch_targets = batch_xy(images, targets)
            pred = model(views)
            loss = mse_loss(pred, batch_targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * views.shape[0]
            count += int(views.shape[0])
        mean_loss = total / max(count, 1)

    model.eval()
    preds: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            views, batch_targets = batch_xy(images, targets)
            preds.append(model(views).detach().cpu())
            tgts.append(batch_targets.detach().cpu())
    pred_cat = torch.cat(preds, dim=0)
    tgt_cat = torch.cat(tgts, dim=0)
    rmse = per_axis_rmse(pred_cat, tgt_cat, y_fields)
    # Rename keys: per_axis_rmse uses train_ prefix already.
    baseline = constant_mean_target_baseline_rmse(tgt_cat)
    collapsed = collapse_to_mean_target(pred_cat, tgt_cat)
    success = overfit_success(
        train_rmse_total=rmse["train_RMSE_total"],
        baseline_rmse=baseline,
    )
    tgt_pair = pairwise_distance_stats(tgt_cat)
    pred_pair = pairwise_distance_stats(pred_cat)
    return {
        "final_train_loss": float(mean_loss),
        **rmse,
        "constant_mean_target_baseline": float(baseline),
        "collapse_to_mean_target": bool(collapsed),
        "overfit_success": bool(success),
        "target_pairwise_mean": tgt_pair["pairwise_mean"],
        "pred_pairwise_mean": pred_pair["pairwise_mean"],
        "target_coords": tgt_cat.tolist(),
        "pred_coords": pred_cat.tolist(),
    }


def run_stage1_overfitability(
    *,
    configs: Sequence[SingleViewArchConfig],
    subsets: Sequence[SharedSubset],
    make_subset_loader: Callable[[SharedSubset], DataLoader],
    batch_xy: BatchXY,
    n_outputs: int,
    y_fields: Sequence[str],
    device: torch.device | str,
    sample_hw: tuple[int, int],
    num_epochs: int = 200,
    lr: float = 1e-4,
    experiment_prefix: str = "win3_stage1",
) -> pd.DataFrame:
    """Stage 1: tiny-subset overfitability across a fixed architecture grid.

    Uses **train subsets only**. Does not touch validation or test data.
    Each architecture × subset cell trains a fresh model and logs
    root-mean-square error (RMSE), baseline comparison, and collapse detection.
    """
    rows: list[dict[str, Any]] = []
    h, w = int(sample_hw[0]), int(sample_hw[1])
    dummy = torch.zeros(1, 1, h, w, device=device, dtype=torch.float32)

    for config in configs:
        for subset in subsets:
            model = build_from_config(config, n_outputs=n_outputs, device=device)
            materialize_lazy_modules(model, dummy)
            n_params = count_parameters(model)
            geom = describe_feature_geometry(model.encoder, height=h, width=w)
            loader = make_subset_loader(subset)
            metrics = train_subset_run(
                model=model,
                loader=loader,
                batch_xy=batch_xy,
                device=device,
                y_fields=y_fields,
                num_epochs=num_epochs,
                lr=lr,
                seed=subset.seed,
            )
            row = {
                "experiment_id": (
                    f"{experiment_prefix}__{config.arch_name}__{subset.subset_id}"
                ),
                "architecture_name": config.arch_name,
                "architecture_config": str(config.to_dict()),
                "parameter_count": int(n_params),
                "flatten_length": int(geom["flatten_length"]),
                "feature_map_hw": str(geom["feature_map_hw"]),
                "input_representation": subset.input_representation,
                "normalisation": subset.normalisation,
                "n": int(subset.n),
                "rep": int(subset.rep),
                "seed": int(subset.seed),
                "subset_id": subset.subset_id,
                "sequence_ids": ",".join(subset.sequence_ids),
                "train_loss_final": metrics["final_train_loss"],
                "train_RMSE_total": metrics["train_RMSE_total"],
                "train_RMSE_X": metrics["train_RMSE_X"],
                "train_RMSE_Y": metrics["train_RMSE_Y"],
                "train_RMSE_Z": metrics["train_RMSE_Z"],
                "constant_mean_target_baseline": metrics[
                    "constant_mean_target_baseline"
                ],
                "collapse_to_mean_target": metrics["collapse_to_mean_target"],
                "overfit_success": metrics["overfit_success"],
                "target_pairwise_mean": metrics["target_pairwise_mean"],
                "pred_pairwise_mean": metrics["pred_pairwise_mean"],
            }
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_stage1(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate overfit rows by architecture and subset size ``n``.

    Collapses repeated ``rep`` draws into mean ± std and success/collapse rates.
    Input is the long-form DataFrame from :func:`run_stage1_overfitability`.

    Args:
        df: Stage-1 results with columns ``architecture_name``,
            ``parameter_count``, ``n``, and metric columns.

    Returns:
        One row per ``(architecture_name, parameter_count, n)`` with
        ``{metric}_mean``, ``{metric}_std``, ``overfit_success_rate``, and
        ``collapse_rate`` when those columns exist. Empty input → empty frame.
    """
    if df.empty:
        return df
    rows: list[dict[str, Any]] = []
    for (arch, n_params, n), g in df.groupby(
        ["architecture_name", "parameter_count", "n"], sort=True
    ):
        row: dict[str, Any] = {
            "architecture_name": arch,
            "parameter_count": int(n_params),
            "n": int(n),
            "n_reps": int(len(g)),
        }
        for col in (
            "train_loss_final",
            "train_RMSE_total",
            "train_RMSE_X",
            "train_RMSE_Y",
            "train_RMSE_Z",
        ):
            if col in g.columns:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=0))
        if "overfit_success" in g.columns:
            row["overfit_success_rate"] = float(g["overfit_success"].mean())
        if "collapse_to_mean_target" in g.columns:
            row["collapse_rate"] = float(g["collapse_to_mean_target"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_split_rmse(
    *,
    model: nn.Module,
    loader: DataLoader,
    batch_xy: BatchXY,
    y_fields: Sequence[str],
    prefix: str,
) -> dict[str, float]:
    """Evaluate root-mean-square error (RMSE) on one split loader (train, validation, or test).

    Keys are prefixed with ``prefix`` (e.g. ``validation_RMSE_total``). Use
    separate loaders per split; never evaluate test during training loops.

    Args:
        model: Trained localiser in eval mode.
        loader: DataLoader for a single split (train, val, or test).
        batch_xy: Maps catalog dict batches to ``(views, targets)``.
        y_fields: Target column order.
        prefix: Split name prepended to RMSE keys (``train``, ``validation``,
            ``test``).

    Returns:
        Dict with ``{prefix}_RMSE_total``, ``{prefix}_RMSE_X/Y/Z``; all
        ``NaN`` when the loader is empty.
    """
    model.eval()
    preds: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            views, batch_targets = batch_xy(images, targets)
            preds.append(model(views).detach().cpu())
            tgts.append(batch_targets.detach().cpu())
    if not preds:
        return {
            f"{prefix}_RMSE_total": float("nan"),
            f"{prefix}_RMSE_X": float("nan"),
            f"{prefix}_RMSE_Y": float("nan"),
            f"{prefix}_RMSE_Z": float("nan"),
        }
    pred_cat = torch.cat(preds, dim=0)
    tgt_cat = torch.cat(tgts, dim=0)
    raw = per_axis_rmse(pred_cat, tgt_cat, y_fields)
    return {
        f"{prefix}_RMSE_total": raw["train_RMSE_total"],
        f"{prefix}_RMSE_X": raw["train_RMSE_X"],
        f"{prefix}_RMSE_Y": raw["train_RMSE_Y"],
        f"{prefix}_RMSE_Z": raw["train_RMSE_Z"],
    }


def train_full_split(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    batch_xy: BatchXY,
    device: torch.device | str,
    num_epochs: int,
    lr: float,
    val_loader: DataLoader | None = None,
    y_fields: Sequence[str] = (),
    early_stop_patience: int | None = None,
    progress_label: str | None = None,
) -> dict[str, Any]:
    """Train on the full train split with optional validation early stopping.

    Never pass the test loader here. Test evaluation is a separate one-shot
    call after architecture / representation / normalisation are fixed.

    When ``progress_label`` is set, prints one line per epoch (train loss and
    optional validation root-mean-square error (RMSE)) with ``flush=True`` for
    long training runs.

    Args:
        model: Localiser to train.
        train_loader: Full train-split DataLoader.
        batch_xy: Maps catalog dict batches to ``(views, targets)``.
        device: Torch device (tensors should already match).
        num_epochs: Maximum epoch budget.
        lr: Adam learning rate (LR).
        val_loader: Optional validation loader for early stopping.
        y_fields: Required when ``val_loader`` is set (for RMSE monitoring).
        early_stop_patience: Stop after this many epochs without val improvement.
        progress_label: Optional prefix for per-epoch stdout lines.

    Returns:
        ``{"history": [...], "best_validation_RMSE_total": float,
        "parameter_count": int}``. History rows contain ``epoch``,
        ``train_loss``, and validation RMSE keys when applicable.
    """
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    stall = 0
    label = str(progress_label).strip() if progress_label else ""

    for epoch in range(int(num_epochs)):
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
            total += float(loss.detach()) * views.shape[0]
            count += int(views.shape[0])
        train_loss = total / max(count, 1)
        record: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
        }
        val_rmse: float | None = None
        improved = False
        if val_loader is not None and y_fields:
            val_metrics = evaluate_split_rmse(
                model=model,
                loader=val_loader,
                batch_xy=batch_xy,
                y_fields=y_fields,
                prefix="validation",
            )
            record.update(val_metrics)
            val_rmse = float(val_metrics["validation_RMSE_total"])
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                stall = 0
                improved = True
            else:
                stall += 1
                if (
                    early_stop_patience is not None
                    and stall >= int(early_stop_patience)
                ):
                    history.append(record)
                    if label:
                        marker = "*" if improved else " "
                        print(
                            f"  {label} epoch {epoch + 1:03d}/{int(num_epochs)}  "
                            f"train_loss={train_loss:.5f}  "
                            f"val_RMSE={val_rmse:.4f}  best={best_val:.4f}{marker}  "
                            f"stall={stall}/{int(early_stop_patience)}  [early stop]",
                            flush=True,
                        )
                    break
        history.append(record)
        if label:
            if val_rmse is not None:
                marker = "*" if improved else " "
                patience_txt = (
                    f"  stall={stall}/{int(early_stop_patience)}"
                    if early_stop_patience is not None
                    else ""
                )
                print(
                    f"  {label} epoch {epoch + 1:03d}/{int(num_epochs)}  "
                    f"train_loss={train_loss:.5f}  "
                    f"val_RMSE={val_rmse:.4f}  best={best_val:.4f}{marker}"
                    f"{patience_txt}",
                    flush=True,
                )
            else:
                print(
                    f"  {label} epoch {epoch + 1:03d}/{int(num_epochs)}  "
                    f"train_loss={train_loss:.5f}",
                    flush=True,
                )

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_validation_RMSE_total": best_val,
        "parameter_count": count_parameters(model),
    }


__all__ = [
    "SharedSubset",
    "build_shared_subsets",
    "collapse_to_mean_target",
    "constant_mean_target_baseline_rmse",
    "evaluate_split_rmse",
    "overfit_success",
    "pairwise_distance_stats",
    "per_axis_rmse",
    "run_stage1_overfitability",
    "summarize_stage1",
    "train_full_split",
    "train_subset_run",
]
