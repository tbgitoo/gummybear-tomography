"""Export M8 Step 3 Fourier weights into a local Hugging Face model clone."""

from __future__ import annotations

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
from tomography_ml_huggingface.hf_local import (
    DEFAULT_LOCAL_TOML,
    HfModelLocalPaths,
    load_hf_local_toml,
    resolve_hf_model_paths,
)
from tomography_ml_huggingface.hub_model_card import (
    CONFIG_NAME,
    DATASET_HUB_ID,
    FINAL_REPORT_URL,
    ModelCardEvalSpec,
    README_NAME,
    REPO_URL,
    WEIGHTS_NAME,
    build_evaluation_results_prose,
    build_model_card_frontmatter,
    build_results_table,
    write_hub_model_artifacts,
)

DEFAULT_STUDY_CHECKPOINT = Path("checkpoints/m8/m08_train_val_test_xyz.pt")
MODEL_KEY = "singleview_cnn_fourier"
ARCH = "fourier"


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


def resolve_singleview_cnn_fourier_paths(
    repo_root: Path,
    *,
    local_toml: Path | None = None,
) -> HfModelLocalPaths:
    """Read ``models.singleview_cnn_fourier`` from gitignored local.toml."""
    return resolve_hf_model_paths(
        repo_root, model_key=MODEL_KEY, local_toml=local_toml
    )


def _metrics_from_study_blob(blob: Mapping[str, Any]) -> dict[str, float]:
    """Pull Fourier held-out **Euclidean** RMSE (+ lr / n_params) from the study blob."""
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
    y_bullets = (
        "\n".join(f"- {y}" for y in y_fields)
        if y_fields
        else "- particle_x\n- particle_y\n- particle_z"
    )
    eval_spec = ModelCardEvalSpec(
        task_name="Single-view particle localisation (xyz)",
        dataset_name="GummyBear Tomography (M8)",
        source_name="Final Report M8 Step 3",
        tags=(
            "tomography",
            "particle-localization",
            "cnn",
            "fourier-pooling",
        ),
    )
    ckpt_rel = str(config.get("source_checkpoint") or DEFAULT_STUDY_CHECKPOINT)
    lines = [
        build_model_card_frontmatter(
            hub_id=hub_id, metrics=metrics, eval_spec=eval_spec
        ).rstrip("\n"),
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
        f"(see {REPO_URL}).",
        "",
        "Single-view CNN + **Fourier pooling** particle localiser from the",
        "gummybear-tomography Final Report **M8 Step 3**",
        "(train → validation/test on `(particle_x, particle_y, particle_z)`).",
        "",
        f"- Hub: [{hub_id}]({hub_url})",
        "- Companion dataset: "
        f"[{DATASET_HUB_ID}](https://huggingface.co/datasets/{DATASET_HUB_ID})",
        "- Runnable report: "
        f"[GummyBearTomography_Final_Report.ipynb]({FINAL_REPORT_URL}) (M8 Step 3)",
        "- Source study checkpoint: "
        f"[{ckpt_rel}]"
        f"(https://huggingface.co/datasets/{DATASET_HUB_ID}/blob/main/"
        f"{ckpt_rel}) (arch `{ARCH}` only)",
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
        *build_evaluation_results_prose(
            eval_spec=eval_spec,
            protocol_lines=[
                "- Protocol: Final Report **M8 Step 3** "
                "(single-view `180°`, `anomaly_ref`, `per_image_zscore`, targets "
                "`particle_x, particle_y, particle_z`)",
            ],
            checkpoint_markdown=(
                "- Source checkpoint: "
                f"[{ckpt_rel}]"
                f"(https://huggingface.co/datasets/{DATASET_HUB_ID}/blob/main/"
                f"{ckpt_rel}) (arch `fourier` only)"
            ),
        ),
        "This matches the Final Report M8 Step 3 bars (not element-wise MSE).",
        "",
        *build_results_table(metrics),
        "Source: "
        f"[Final Report M8 Step 3]({FINAL_REPORT_URL}).",
        "",
        "## Load",
        "",
        "```python",
        "import torch",
        f"# libraries from {REPO_URL}",
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
        f"({REPO_URL}/blob/master/"
        "notebooks/milestone_11/11_1_test_singleview_cnn_fourier.ipynb) "
        f"in the [{REPO_URL.split('//')[-1]}]({REPO_URL}) "
        "repository.",
        "",
    ]
    return "\n".join(lines)


def export_singleview_cnn_fourier(
    repo_root: Path,
    *,
    local_toml: Path | None = None,
    checkpoint_path: Path | None = None,
    local_clone: Path | None = None,
    image_hw: tuple[int, int] = (128, 128),
) -> M8FourierExportResult:
    """Extract M8 Step 3 Fourier weights into the local Hub model clone."""
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
    y_fields = [
        str(y)
        for y in (blob.get("y_fields") or ("particle_x", "particle_y", "particle_z"))
    ]
    x_field = str(blob.get("x_field") or "anomaly_ref")
    metrics = _metrics_from_study_blob(blob)

    freeze = m8_single_view_block_freeze()
    h, w = int(image_hw[0]), int(image_hw[1])
    model = make_m8_single_view_model(ARCH, n_outputs=len(y_fields), device="cpu")
    materialize_lazy_modules(model, torch.zeros(1, 1, h, w))
    model.load_state_dict(state, strict=True)
    n_params = int(count_parameters(model))

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
    weights_path, config_path, readme_path = write_hub_model_artifacts(
        dest,
        state_dict=model.state_dict(),
        config=config,
        readme_markdown=_model_card_markdown(
            hub_id=paths.hub_id,
            hub_url=paths.hub_url,
            config=config,
            metrics=metrics,
        ),
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
