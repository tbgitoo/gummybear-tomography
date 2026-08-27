"""Export selected localisation weights into a local Hugging Face model clone."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from gummybear.paths import display_path
from tomography_ml.localization.builders import (
    count_parameters,
    m8_single_view_block_freeze,
    materialize_lazy_modules,
)
from tomography_ml.studies.single_view_m8 import make_m8_single_view_model

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover — Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_LOCAL_TOML = Path("configs/hf/local.toml")
DEFAULT_STUDY_CHECKPOINT = Path("checkpoints/m8/m08_train_val_test_xyz.pt")
MODEL_KEY = "singleview_cnn_fourier"
ARCH = "fourier"
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"
README_NAME = "README.md"


@dataclass(frozen=True)
class HfModelLocalPaths:
    """Resolved Hub id + local clone path from ``configs/hf/local.toml``."""

    hub_id: str
    hub_url: str
    local_clone: Path


@dataclass(frozen=True)
class M8FourierExportResult:
    """Artefacts written for ``tbhugging/singleview_cnn_fourier``."""

    hub_id: str
    local_clone: Path
    checkpoint_path: Path
    weights_path: Path
    config_path: Path
    readme_path: Path
    n_params: int
    lr: float
    metrics: dict[str, float]


def load_hf_local_toml(path: Path) -> dict[str, Any]:
    """Parse a machine-local Hugging Face staging TOML."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {display_path(path)}. Copy configs/hf/local.toml.example "
            "to configs/hf/local.toml and set models.singleview_cnn_fourier.local_clone."
        )
    with path.open("rb") as fh:
        return tomllib.load(fh)


def resolve_singleview_cnn_fourier_paths(
    repo_root: Path,
    *,
    local_toml: Path | None = None,
) -> HfModelLocalPaths:
    """Read ``models.singleview_cnn_fourier`` from gitignored local.toml."""
    repo_root = Path(repo_root).resolve()
    toml_path = (
        Path(local_toml)
        if local_toml is not None
        else repo_root / DEFAULT_LOCAL_TOML
    )
    if not toml_path.is_absolute():
        toml_path = repo_root / toml_path
    data = load_hf_local_toml(toml_path)
    models = data.get("models") or {}
    row = models.get(MODEL_KEY)
    if not isinstance(row, Mapping):
        raise KeyError(
            f"{display_path(toml_path)} missing [models.{MODEL_KEY}] table"
        )
    hub_id = str(row.get("hub_id") or "").strip()
    hub_url = str(row.get("hub_url") or "").strip()
    local_clone = Path(str(row.get("local_clone") or "").strip()).expanduser()
    if not hub_id:
        raise ValueError(f"[models.{MODEL_KEY}] hub_id is required")
    if not local_clone.is_absolute():
        local_clone = (repo_root / local_clone).resolve()
    if not hub_url:
        hub_url = f"https://huggingface.co/{hub_id}"
    return HfModelLocalPaths(
        hub_id=hub_id, hub_url=hub_url, local_clone=local_clone
    )


def _metrics_from_study_blob(blob: Mapping[str, Any]) -> dict[str, float]:
    """Pull Fourier held-out **Euclidean** RMSE (+ lr / n_params) from the study blob.

    M8 stores two different reductions of the residual:

    - ``*_MSE``: element-wise mean over the three coordinates (training loss scale).
    - ``*_RMSE``: RMSE of per-sample L2 norms
      ``d_i = ||pred_i - y_i||_2``, i.e.
      ``sqrt(mean_i d_i^2)`` — the report's localisation error.

    For three targets these are related by ``RMSE_L2 ≈ sqrt(3) * sqrt(MSE)``,
    so publishing both unlabeled is misleading. Hub export keeps only L2 RMSE.
    """
    out: dict[str, float] = {}
    comparison = blob.get("comparison")
    if not isinstance(comparison, Mapping):
        return out
    variants = list(comparison.get("backbone_kind") or comparison.get("variant_id") or [])
    idx = None
    for i, v in enumerate(variants):
        if str(v).lower() in {ARCH, f"m08_3a2xyz_{ARCH}"} or str(v).endswith(
            f"_{ARCH}"
        ):
            idx = i
            break
    if idx is None:
        for i, v in enumerate(variants):
            if ARCH in str(v).lower():
                idx = i
                break
    if idx is None:
        return out
    # Prefer explicit Euclidean RMSE names in the exported metrics dict.
    key_map = {
        "validation_RMSE": "validation_RMSE_total",
        "test_RMSE": "test_RMSE_total",
        "n_params": "n_params",
        "lr": "lr",
    }
    for src, dst in key_map.items():
        series = comparison.get(src)
        if series is None:
            continue
        try:
            out[dst] = float(series[idx])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _model_card_frontmatter(
    *,
    hub_id: str,
    metrics: Mapping[str, float],
) -> str:
    """Build Hub-compliant YAML metadata (pipeline_tag, metrics, model-index)."""
    try:
        from huggingface_hub import EvalResult, ModelCardData
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to write the Hub model card. "
            'Install with: pip install ".[hf]" -c requirements.txt'
        ) from exc

    source_url = (
        "https://github.com/tbgitoo/gummybear-tomography/blob/master/"
        "GummyBearTomography_Final_Report.ipynb"
    )
    source_name = "Final Report M8 Step 3"
    task_type = "image-feature-extraction"
    task_name = "Single-view particle localisation (xyz)"
    dataset_type = "tbhugging/gummybear-tomography"
    dataset_name = "GummyBear Tomography (M8)"
    metric_type = "rmse"
    metric_name = "RMSE_total (Euclidean xyz)"

    eval_results: list[Any] = []
    for split_key, split_name in (
        ("validation_RMSE_total", "validation"),
        ("test_RMSE_total", "test"),
    ):
        if split_key not in metrics:
            continue
        eval_results.append(
            EvalResult(
                task_type=task_type,
                task_name=task_name,
                dataset_type=dataset_type,
                dataset_name=dataset_name,
                dataset_config="m8_1",
                dataset_split=split_name,
                metric_type=metric_type,
                metric_name=metric_name,
                metric_value=round(float(metrics[split_key]), 6),
                source_name=source_name,
                source_url=source_url,
            )
        )

    card_data = ModelCardData(
        license="apache-2.0",
        library_name="pytorch",
        pipeline_tag=task_type,
        tags=[
            "tomography",
            "particle-localization",
            "cnn",
            "fourier-pooling",
        ],
        datasets=[dataset_type],
        metrics=[metric_type],
        model_name=hub_id,
        eval_results=eval_results or None,
    )
    yaml_body = card_data.to_yaml().rstrip() + "\n"
    return f"---\n{yaml_body}---\n"


def _model_card_markdown(
    *,
    hub_id: str,
    hub_url: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> str:
    freeze = m8_single_view_block_freeze()
    h = int(config.get("image_height") or 128)
    w = int(config.get("image_width") or 128)
    y_fields = [str(y) for y in (config.get("y_fields") or ())]
    y_bullets = "\n".join(f"- {y}" for y in y_fields) if y_fields else "- particle_x\n- particle_y\n- particle_z"
    lines = [
        _model_card_frontmatter(hub_id=hub_id, metrics=metrics).rstrip("\n"),
        "",
        f"# `{hub_id}`",
        "",
        "## What this model does",
        "",
        f"Given a single {h}×{w} projection image from the M8",
        "GummyBear Tomography dataset, the model predicts",
        "",
        y_bullets,
        "",
        "coordinates of the embedded particle.",
        "",
        "",
        "## Architecture",
        "",
        "The model uses Fourier pooling in place of conventional",
        "max-pooling layers as described in the M8 study "
        "(see https://github.com/tbgitoo/gummybear-tomography).",
        "",
        "Single-view CNN + **Fourier pooling** particle localiser from the",
        "gummybear-tomography Final Report **M8 Step 3**",
        "(train → validation/test on `(particle_x, particle_y, particle_z)`).",
        "",
        f"- Hub: [{hub_id}]({hub_url})",
        "- Companion dataset: "
        "[tbhugging/gummybear-tomography](https://huggingface.co/datasets/tbhugging/gummybear-tomography)",
        "- Runnable report: "
        "[GummyBearTomography_Final_Report.ipynb]"
        "(https://github.com/tbgitoo/gummybear-tomography/blob/master/"
        "GummyBearTomography_Final_Report.ipynb) (M8 Step 3)",
        "- Source study checkpoint: "
        "[checkpoints/m8/m08_train_val_test_xyz.pt]"
        "(https://huggingface.co/datasets/tbhugging/gummybear-tomography/blob/main/"
        f"checkpoints/m8/m08_train_val_test_xyz.pt) (arch `{ARCH}` only)",
        "",
        "## Training configuration",
        "",
        f"- Input field: `{config['x_field']}`",
        f"- Normalisation: `{config['image_normalize']}`",
        f"- Camera: `{config['keep_angles_deg']}`° (single view)",
        f"- Targets: `{', '.join(config['y_fields'])}`",
        f"- Library class: `{freeze.library_class}`",
        f"- Trainable parameters: `{config['n_params']}`",
        f"- Stage learning rate: `{config['lr']}`",
        "",
        "## Evaluation Results",
        "",
        "Structured scores for the Hub widget are declared in the YAML "
        "`model-index` / `metrics` metadata "
        "([Model Cards — Evaluation Results]"
        "(https://huggingface.co/docs/hub/model-cards#evaluation-results)).",
        "",
        "### Testing Data",
        "",
        "- Dataset: "
        "[tbhugging/gummybear-tomography]"
        "(https://huggingface.co/datasets/tbhugging/gummybear-tomography) "
        "(config `m8_1`)",
        "- Splits: `validation`, `test`",
        "- Protocol: Final Report **M8 Step 3** "
        "(single-view `180°`, `anomaly_ref`, `per_image_zscore`, targets "
        "`particle_x, particle_y, particle_z`)",
        "- Source checkpoint: "
        "[checkpoints/m8/m08_train_val_test_xyz.pt]"
        "(https://huggingface.co/datasets/tbhugging/gummybear-tomography/blob/main/"
        "checkpoints/m8/m08_train_val_test_xyz.pt) (arch `fourier` only)",
        "",
        "### Metrics",
        "",
        "Reported error is **Euclidean RMSE** over particle `(x,y,z)`:",
        "`d_i = ||pred_i - y_i||_2`, then `RMSE_total = sqrt(mean_i d_i^2)`.",
        "This matches the Final Report M8 Step 3 bars (not element-wise MSE).",
        "Hub metric id: `rmse` (display name `RMSE_total (Euclidean xyz)`).",
        "",
        "### Results",
        "",
    ]
    result_rows: list[str] = []
    for key, split_name in (
        ("validation_RMSE_total", "validation"),
        ("test_RMSE_total", "test"),
    ):
        if key in metrics:
            result_rows.append(
                f"| `{split_name}` | RMSE_total (Euclidean xyz) | "
                f"`{metrics[key]:.6f}` |"
            )
    if result_rows:
        lines.extend(
            [
                "| Split | Metric | Value |",
                "| --- | --- | ---: |",
                *result_rows,
                "",
            ]
        )
    lines.extend(
        [
            "Source: "
            "[Final Report M8 Step 3]"
            "(https://github.com/tbgitoo/gummybear-tomography/blob/master/"
            "GummyBearTomography_Final_Report.ipynb).",
            "",
            "## Load",
            "",
            "```python",
            "import torch",
            "# libraries from https://github.com/tbgitoo/gummybear-tomography",
            "from tomography_ml.studies.single_view_m8 import make_m8_single_view_model",
            "from tomography_ml.localization.builders import materialize_lazy_modules",
            "",
            "model = make_m8_single_view_model('fourier', n_outputs=3, device='cpu')",
            f"materialize_lazy_modules(model, torch.zeros(1, 1, {h}, {w}))",
            f"state = torch.load('{WEIGHTS_NAME}', map_location='cpu', weights_only=True)",
            "model.load_state_dict(state)",
            "model.eval()",
            "```",
            "",
            "Do not treat this checkpoint as multi-view (M9) or multi-illumination (M10).",
            "",
            "## Inference",
            "",
            "For an example with worked download, model instanciation and inference, see: "
            "[11_1_test_singleview_cnn_fourier.ipynb]"
            "(https://github.com/tbgitoo/gummybear-tomography/blob/master/"
            "notebooks/milestone_11/11_1_test_singleview_cnn_fourier.ipynb) "
            "in the [gummybear-tomography](https://github.com/tbgitoo/gummybear-tomography) "
            "repository.",
            "",
        ]
    )
    return "\n".join(lines)


def export_singleview_cnn_fourier(
    repo_root: Path,
    *,
    local_toml: Path | None = None,
    checkpoint_path: Path | None = None,
    local_clone: Path | None = None,
    image_hw: tuple[int, int] = (128, 128),
) -> M8FourierExportResult:
    """Extract M8 Step 3 Fourier weights into the local Hub model clone.

    Reads staging path from gitignored ``configs/hf/local.toml`` unless
    ``local_clone`` is passed explicitly. Writes ``pytorch_model.bin``,
    ``config.json``, and ``README.md`` (model card).
    """
    repo_root = Path(repo_root).resolve()
    paths = resolve_singleview_cnn_fourier_paths(repo_root, local_toml=local_toml)
    dest = Path(local_clone).expanduser() if local_clone is not None else paths.local_clone
    if not dest.is_absolute():
        dest = (repo_root / dest).resolve()

    ckpt = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else repo_root / DEFAULT_STUDY_CHECKPOINT
    )
    if not ckpt.is_absolute():
        ckpt = repo_root / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Missing study checkpoint {display_path(ckpt)} "
            "(run Final Report M8 Step 3 / train_val_test xyz first)."
        )

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "final_state_by_arch" not in blob or ARCH not in blob["final_state_by_arch"]:
        raise KeyError(
            f"{display_path(ckpt)} missing final_state_by_arch[{ARCH!r}]"
        )
    state = blob["final_state_by_arch"][ARCH]
    lr = float((blob.get("lr_by_arch") or {}).get(ARCH, 0.03))
    y_fields = [str(y) for y in (blob.get("y_fields") or ("particle_x", "particle_y", "particle_z"))]
    x_field = str(blob.get("x_field") or "anomaly_ref")
    metrics = _metrics_from_study_blob(blob)

    freeze = m8_single_view_block_freeze()
    h, w = int(image_hw[0]), int(image_hw[1])
    model = make_m8_single_view_model(ARCH, n_outputs=len(y_fields), device="cpu")
    materialize_lazy_modules(model, torch.zeros(1, 1, h, w))
    model.load_state_dict(state, strict=True)
    n_params = int(count_parameters(model))

    dest.mkdir(parents=True, exist_ok=True)
    weights_path = dest / WEIGHTS_NAME
    torch.save(model.state_dict(), weights_path)

    config: dict[str, Any] = {
        "model_type": MODEL_KEY,
        "hub_id": paths.hub_id,
        "architecture": ARCH,
        "library_class": freeze.library_class,
        "head_type": "linear",
        "n_outputs": len(y_fields),
        "y_fields": y_fields,
        "x_field": x_field,
        "image_normalize": freeze.image_normalize,
        "representation": freeze.representation_name,
        "keep_angles_deg": float(freeze.keep_angles_deg),
        "input_channels": 1,
        "image_height": h,
        "image_width": w,
        "lr": float(metrics.get("lr", lr)),
        "n_params": int(metrics.get("n_params", n_params)),
        "source_checkpoint": "checkpoints/m8/m08_train_val_test_xyz.pt",
        "source_experiment_id": "m08_3a2_train_validation_study_xyz",
        "weights_file": WEIGHTS_NAME,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    config_path = dest / CONFIG_NAME
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    readme_path = dest / README_NAME
    readme_path.write_text(
        _model_card_markdown(
            hub_id=paths.hub_id,
            hub_url=paths.hub_url,
            config=config,
            metrics=metrics,
        ),
        encoding="utf-8",
    )

    return M8FourierExportResult(
        hub_id=paths.hub_id,
        local_clone=dest,
        checkpoint_path=ckpt,
        weights_path=weights_path,
        config_path=config_path,
        readme_path=readme_path,
        n_params=n_params,
        lr=float(config["lr"]),
        metrics={k: float(v) for k, v in metrics.items()},
    )


__all__ = [
    "ARCH",
    "CONFIG_NAME",
    "DEFAULT_LOCAL_TOML",
    "DEFAULT_STUDY_CHECKPOINT",
    "HfModelLocalPaths",
    "M8FourierExportResult",
    "MODEL_KEY",
    "README_NAME",
    "WEIGHTS_NAME",
    "export_singleview_cnn_fourier",
    "load_hf_local_toml",
    "resolve_singleview_cnn_fourier_paths",
]
