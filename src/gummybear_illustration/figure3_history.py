"""JSON prediction-history schema for Figure 3 (M8 pooling vs Fourier)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
MODEL_POOLING = "m8_pooling"
MODEL_FOURIER = "fourier_pooling"
VIEW_MODE = "single-view"
COORDINATE_SYSTEM = "simulation_mm_z_up"
COORDINATE_TRANSFORM = "identity"

# Catalog / STL millimetres are already the POV-Ray world frame (z-up).
# Predicted and ground-truth xyz are written and rendered without remapping.


def default_figure3_root(repo: Path) -> Path:
    """Gitignored illustration output: ``outputs/figure3_learning_convergence``."""
    return Path(repo) / "outputs" / "figure3_learning_convergence"


def history_paths(root: Path) -> dict[str, Path]:
    hist = Path(root) / "prediction_history"
    return {
        "root": hist,
        "pooling": hist / "m8_pooling_history.json",
        "fourier": hist / "fourier_pooling_history.json",
        "combined": hist / "combined_prediction_history.json",
    }


def empty_history(
    *,
    model_type: str,
    arch: str,
    lr: float,
    seed: int,
    num_epochs: int,
    y_fields: Sequence[str],
    tracked_samples: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": str(model_type),
        "arch": str(arch),
        "view_mode": VIEW_MODE,
        "lr": float(lr),
        "seed": int(seed),
        "num_epochs": int(num_epochs),
        "y_fields": [str(y) for y in y_fields],
        "coordinate_system": COORDINATE_SYSTEM,
        "coordinate_transform": COORDINATE_TRANSFORM,
        "coordinate_transform_note": (
            "particle_x/y/z from the M8 catalog are simulation millimetres "
            "(z-up), identical to the STL / physical-scene POV world. "
            "No extra transform is applied when logging or rendering."
        ),
        "tracked_samples": [dict(s) for s in tracked_samples],
        "records": [],
        "extra": dict(extra or {}),
    }


def append_record(
    history: dict[str, Any],
    *,
    epoch: int,
    sample_id: str,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    localization_error: float,
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> None:
    true = [float(v) for v in y_true]
    pred = [float(v) for v in y_pred]
    history["records"].append(
        {
            "model_type": str(history["model_type"]),
            "epoch": int(epoch),
            "sample_id": str(sample_id),
            "view_mode": VIEW_MODE,
            "y_true": true,
            "y_pred": pred,
            "localization_error": float(localization_error),
            "train_loss": None if train_loss is None else float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
        }
    )


def save_history(path: str | Path, history: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_history(path: str | Path) -> dict[str, Any]:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(blob, dict):
        raise TypeError(f"history must be a JSON object; got {type(blob)!r}")
    if int(blob.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version={blob.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return blob


def combine_histories(
    pooling: Mapping[str, Any],
    fourier: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "view_mode": VIEW_MODE,
        "coordinate_system": COORDINATE_SYSTEM,
        "coordinate_transform": COORDINATE_TRANSFORM,
        "coordinate_transform_note": pooling.get(
            "coordinate_transform_note",
            fourier.get("coordinate_transform_note", ""),
        ),
        "y_fields": list(pooling.get("y_fields") or fourier.get("y_fields") or []),
        "seed": pooling.get("seed", fourier.get("seed")),
        "num_epochs": max(
            int(pooling.get("num_epochs", 0)),
            int(fourier.get("num_epochs", 0)),
        ),
        "tracked_samples": list(
            pooling.get("tracked_samples") or fourier.get("tracked_samples") or []
        ),
        "models": {
            MODEL_POOLING: {
                "arch": pooling.get("arch"),
                "lr": pooling.get("lr"),
                "extra": pooling.get("extra", {}),
            },
            MODEL_FOURIER: {
                "arch": fourier.get("arch"),
                "lr": fourier.get("lr"),
                "extra": fourier.get("extra", {}),
            },
        },
        "records": list(pooling.get("records") or [])
        + list(fourier.get("records") or []),
    }


@dataclass(frozen=True)
class Figure3RecordIndex:
    """Combined-history records keyed by ``(sample_id, model_type)``."""

    by_sample_model: Mapping[tuple[str, str], tuple[dict[str, Any], ...]]

    @classmethod
    def from_combined(cls, combined: Mapping[str, Any]) -> Figure3RecordIndex:
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for rec in combined.get("records") or []:
            sid = str(rec.get("sample_id", ""))
            mt = str(rec.get("model_type", ""))
            if not sid or not mt:
                continue
            buckets.setdefault((sid, mt), []).append(dict(rec))
        frozen = {
            key: tuple(
                sorted(
                    recs,
                    key=lambda r: (int(r["epoch"]), str(r.get("model_type", ""))),
                )
            )
            for key, recs in buckets.items()
        }
        return cls(by_sample_model=frozen)

    def records_for_sample(
        self,
        sample_id: str,
        *,
        model_type: str,
    ) -> list[dict[str, Any]]:
        recs = self.by_sample_model.get((str(sample_id), str(model_type)))
        return list(recs) if recs else []


def build_figure3_record_index(combined: Mapping[str, Any]) -> Figure3RecordIndex:
    """Build a lookup table for ``records_for_sample`` on a combined history."""
    return Figure3RecordIndex.from_combined(combined)


def records_for_sample(
    history: Mapping[str, Any],
    sample_id: str,
    *,
    model_type: str | None = None,
    record_index: Figure3RecordIndex | None = None,
) -> list[dict[str, Any]]:
    sid = str(sample_id)
    if record_index is not None:
        if model_type is None:
            out: list[dict[str, Any]] = []
            for mt in (MODEL_POOLING, MODEL_FOURIER):
                out.extend(record_index.records_for_sample(sid, model_type=mt))
            out.sort(key=lambda r: (int(r["epoch"]), str(r.get("model_type", ""))))
            return out
        return record_index.records_for_sample(sid, model_type=str(model_type))
    out: list[dict[str, Any]] = []
    for rec in history.get("records") or []:
        if str(rec.get("sample_id")) != sid:
            continue
        if model_type is not None and str(rec.get("model_type")) != str(model_type):
            continue
        out.append(dict(rec))
    out.sort(key=lambda r: (int(r["epoch"]), str(r.get("model_type", ""))))
    return out


def final_error_by_sample(
    history: Mapping[str, Any],
    *,
    model_type: str,
) -> dict[str, float]:
    last: dict[str, tuple[int, float]] = {}
    for rec in history.get("records") or []:
        if str(rec.get("model_type")) != str(model_type):
            continue
        sid = str(rec["sample_id"])
        epoch = int(rec["epoch"])
        err = float(rec["localization_error"])
        prev = last.get(sid)
        if prev is None or epoch >= prev[0]:
            last[sid] = (epoch, err)
    return {sid: err for sid, (_e, err) in last.items()}


def select_best_fourier_advantage_sample(
    combined: Mapping[str, Any],
) -> tuple[str, float, float, float]:
    """Return ``(sample_id, pooled_err, fourier_err, advantage)``.

    Advantage is ``pooled_final_error - fourier_final_error`` (larger = clearer
    Fourier win on that validation particle).
    """
    pooled = final_error_by_sample(combined, model_type=MODEL_POOLING)
    fourier = final_error_by_sample(combined, model_type=MODEL_FOURIER)
    best_id = ""
    best_adv = float("-inf")
    best_p = float("nan")
    best_f = float("nan")
    for sid, p_err in pooled.items():
        if sid not in fourier:
            continue
        f_err = fourier[sid]
        adv = float(p_err) - float(f_err)
        if adv > best_adv:
            best_adv = adv
            best_id = sid
            best_p = float(p_err)
            best_f = float(f_err)
    if not best_id:
        raise ValueError("combined history has no overlapping tracked samples")
    return best_id, best_p, best_f, best_adv
