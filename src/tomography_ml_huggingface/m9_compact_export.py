"""Export M9 Step 2 / 09_2B compact pooled model into a local Hugging Face clone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from gummybear.paths import display_path
from tomography_ml.localization.builders import (
    count_parameters,
    m8_single_view_block_freeze,
    materialize_lazy_modules,
)
from tomography_ml.localization.localize_multiview import (
    FUSION_PATTERN_09_2_POOLED,
    GeometryAwareFourierFusionLocalizer,
)
from tomography_ml.studies.study_checkpoints import M09_E2E_POOLED_GEOMETRY_FUSION
from tomography_ml_huggingface.hf_local import (
    HfModelLocalPaths,
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

DEFAULT_STUDY_CHECKPOINT = Path("checkpoints/m9") / M09_E2E_POOLED_GEOMETRY_FUSION
FOURIER_STUDY_CHECKPOINT = Path("checkpoints/m9/m09_e2e_fourier_geometry_fusion.pt")
MODEL_KEY = "camera_orbit_compact_09_2b"
PROTOCOL = "09_2B"
ARCHITECTURE = "pooled_gap"
VARIANT_ID = "m09_2_e2e_pooled_geometry_fusion"
STATE_KEY = "f2_state"
Y_FIELDS = ("particle_x", "particle_y", "particle_z")


@dataclass(frozen=True)
class M9CompactExportResult:
    """Artefacts written for ``tbhugging/camera_orbit_compact_09_2b``."""

    hub_id: str
    local_clone: Path
    checkpoint_path: Path
    weights_path: Path
    config_path: Path
    readme_path: Path
    n_params: int
    lr: float
    metrics: dict[str, float]


def resolve_camera_orbit_compact_09_2b_paths(
    repo_root: Path,
    *,
    local_toml: Path | None = None,
) -> HfModelLocalPaths:
    """Read ``models.camera_orbit_compact_09_2b`` from gitignored local.toml."""
    return resolve_hf_model_paths(
        repo_root, model_key=MODEL_KEY, local_toml=local_toml
    )


def _metrics_from_m9_comparison(
    comparison: Any,
    *,
    variant_id: str = VARIANT_ID,
) -> dict[str, float]:
    """Pull compact-variant Euclidean RMSE (+ n_params) from the M9 comparison table."""
    import pandas as pd

    df = comparison if isinstance(comparison, pd.DataFrame) else pd.DataFrame(comparison)
    out: dict[str, float] = {}
    sub = df[df["variant_id"].astype(str) == variant_id]
    for split in ("validation", "test"):
        row = sub[sub["split"].astype(str) == split]
        if row.empty:
            continue
        out[f"{split}_RMSE_total"] = float(row["RMSE_total"].iloc[0])
    if not sub.empty and "learned_parameter_count" in sub.columns:
        out["n_params"] = float(sub["learned_parameter_count"].iloc[0])
    return out


def _format_angle_list(angles_deg: Sequence[float]) -> str:
    return ", ".join(f"{float(a):g}°" for a in angles_deg)


def _assert_pooled_09_2b_checkpoint(blob: Mapping[str, Any], ckpt: Path) -> None:
    """Reject Fourier (09_2A) or other non-pooled study blobs."""
    family = str(blob.get("family") or "").strip().lower()
    if family and family != "pooled":
        raise ValueError(
            f"{display_path(ckpt)} is family={family!r}; "
            f"{PROTOCOL} export requires the pooled (GAP) checkpoint "
            f"{display_path(DEFAULT_STUDY_CHECKPOINT)}, not the Fourier "
            f"({FOURIER_STUDY_CHECKPOINT.name}) study."
        )
    if str(blob.get("experiment_id") or "").endswith("fourier_geometry_fusion"):
        raise ValueError(
            f"{display_path(ckpt)} is the 09_2A Fourier e2e study; "
            f"use {display_path(DEFAULT_STUDY_CHECKPOINT)} for {PROTOCOL} GAP export."
        )


def _model_card_markdown(
    *,
    hub_id: str,
    hub_url: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> str:
    h = int(config.get("image_height") or 128)
    w = int(config.get("image_width") or 128)
    n_views = int(config["n_views"])
    angles = [float(a) for a in config["view_angles_deg"]]
    y_fields = [str(y) for y in config["y_fields"]]
    y_bullets = "\n".join(f"- {y}" for y in y_fields)
    eval_spec = ModelCardEvalSpec(
        task_name="Multi-view camera-orbit particle localisation (xyz)",
        dataset_name="GummyBear Tomography (M8 catalog, M9 multi-view)",
        source_name=f"Final Report M9 Step 2 / {PROTOCOL} (pooled GAP)",
        tags=(
            "tomography",
            "particle-localization",
            "cnn",
            "pooled",
            "gap",
            "multiview",
            "geometry-fusion",
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
        f"Given **{n_views}** camera-orbit views "
        f"({ _format_angle_list(angles) }) of {h}×{w} projections "
        "from the GummyBear Tomography dataset, the model predicts",
        "",
        y_bullets,
        "",
        "coordinates of the embedded particle.",
        "",
        "",
        "## Architecture",
        "",
        "Per-view **GAP (pooled) CNN trunk** → camera **sin/cos geometry tokens** →",
        "**compact fusion MLP** (`e2e_pooled_geometry_fusion`).",
        "",
        f"Final Report **M9 Step 2 / {PROTOCOL}** — compact head (`{STATE_KEY}`).",
        "",
        f"- Hub: [{hub_id}]({hub_url})",
        "- Companion dataset: "
        f"[{DATASET_HUB_ID}](https://huggingface.co/datasets/{DATASET_HUB_ID})",
        "- Runnable report: "
        f"[GummyBearTomography_Final_Report.ipynb]({FINAL_REPORT_URL}) "
        f"(M9 Step 2 — {PROTOCOL} pooled ladder)",
        "- Source study checkpoint: "
        f"[{ckpt_rel}]"
        f"(https://huggingface.co/datasets/{DATASET_HUB_ID}/blob/main/"
        f"{ckpt_rel}) (`{STATE_KEY}` / `{VARIANT_ID}`)",
        "",
        "## Training configuration",
        "",
        f"- Protocol: `{config['protocol']}` (pooled GAP; excludes 09_2A Fourier)",
        f"- Backbone: `{config['architecture']}` (`{config['backbone_kind']}` trunk)",
        f"- Input field: `{config['x_field']}`",
        f"- Normalisation: `{config['image_normalize']}`",
        f"- Camera orbit: `{_format_angle_list(angles)}` ({n_views} views)",
        f"- Geometry: `{', '.join(config['geometry_features'])}` ({config['geometry_mode']})",
        f"- Fusion: hidden `{config['fusion_hidden']}`, depth `{config['fusion_depth']}`",
        f"- Targets: `{', '.join(y_fields)}`",
        f"- Builder: `{config['library_class']}.{config['factory_method']}()`",
        f"- Variant: `{config['variant_id']}`",
        f"- Trainable parameters: `{config['n_params']}`",
        f"- Stage-A learning rate: `{config['lr']}`",
        "",
        *build_evaluation_results_prose(
            eval_spec=eval_spec,
            protocol_lines=[
                f"- Protocol: Final Report **M9 Step 2 / {PROTOCOL}** "
                f"(camera orbit {_format_angle_list(angles)}, "
                "`anomaly_ref`, `per_image_zscore`, **GAP pooled trunk only**, "
                "compact geometry fusion; excludes 09_2A Fourier)",
            ],
            checkpoint_markdown=(
                "- Source checkpoint: "
                f"[{ckpt_rel}]"
                f"(https://huggingface.co/datasets/{DATASET_HUB_ID}/blob/main/"
                f"{ckpt_rel}) (`{STATE_KEY}` only)"
            ),
        ),
        "Scores match the Final Report M9 Step 2 **09_2B pooled GAP** bar "
        "(not 09_2A Fourier; not element-wise MSE).",
        "",
        *build_results_table(metrics),
        "Source: "
        f"[Final Report M9 Step 2 / {PROTOCOL}]({FINAL_REPORT_URL}).",
        "",
        "## Load",
        "",
        "```python",
        "import torch",
        f"# libraries from {REPO_URL}",
        "# Historical class name — use .for_09_2_pooled() for this GAP checkpoint only.",
        "from tomography_ml.localization.localize_multiview import (",
        "    GeometryAwareFourierFusionLocalizer,",
        ")",
        "",
        f"n_views = {n_views}",
        f"view_angles_deg = {list(angles)}",
        "model = GeometryAwareFourierFusionLocalizer.for_09_2_pooled(",
        "    n_views=n_views,",
        "    view_angles_deg=view_angles_deg,",
        ")",
        f"views = torch.zeros(1, n_views, 1, {h}, {w})",
        "model(views)  # materialise lazy layers",
        f"state = torch.load('{WEIGHTS_NAME}', map_location='cpu', weights_only=True)",
        "model.load_state_dict(state)",
        "model.eval()",
        "xyz = model(views)",
        "```",
        "",
        "Input tensor shape: `[batch, n_views, channels, height, width]`.",
        f"Do **not** load with `.for_09_2()` (that builds the 09_2A Fourier trunk).",
        "Also excludes single-view M8, 09_3 large fusion, and M10 illumination stacks.",
        "",
        "## Inference",
        "",
        "For an example with worked download, model instanciation and inference, see: "
        "[11_2_test_camera_orbit_compact_09_2b.ipynb]"
        f"({REPO_URL}/blob/master/"
        "notebooks/milestone_11/11_2_test_camera_orbit_compact_09_2b.ipynb) "
        f"in the [{REPO_URL.split('//')[-1]}]({REPO_URL}) repository.",
        "",
    ]
    return "\n".join(lines)


def export_camera_orbit_compact_09_2b(
    repo_root: Path,
    *,
    local_toml: Path | None = None,
    checkpoint_path: Path | None = None,
    local_clone: Path | None = None,
    image_hw: tuple[int, int] = (128, 128),
) -> M9CompactExportResult:
    """Extract M9 09_2B compact pooled weights into the local Hub model clone."""
    repo_root = Path(repo_root).resolve()
    paths = resolve_camera_orbit_compact_09_2b_paths(
        repo_root, local_toml=local_toml
    )
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
            "(run Final Report M9 Step 2 / 09_2B pooled e2e first)."
        )

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    _assert_pooled_09_2b_checkpoint(blob, ckpt)
    if STATE_KEY not in blob:
        raise KeyError(f"{display_path(ckpt)} missing {STATE_KEY!r}")
    state = blob[STATE_KEY]
    view_angles_deg = [float(a) for a in blob.get("view_angles_deg") or ()]
    if not view_angles_deg:
        raise KeyError(f"{display_path(ckpt)} missing view_angles_deg")
    n_views = len(view_angles_deg)
    lr = float(blob.get("lr_stage_a") or 0.001)
    metrics = _metrics_from_m9_comparison(blob.get("comparison"))
    if "lr" not in metrics:
        metrics = {**metrics, "lr": lr}

    freeze = m8_single_view_block_freeze()
    h, w = int(image_hw[0]), int(image_hw[1])
    model = GeometryAwareFourierFusionLocalizer.for_09_2_pooled(
        n_views=n_views,
        view_angles_deg=view_angles_deg,
    )
    materialize_lazy_modules(model, torch.zeros(1, n_views, 1, h, w))
    model.load_state_dict(state, strict=True)
    n_params = int(count_parameters(model))

    config: dict[str, Any] = {
        "model_type": MODEL_KEY,
        "hub_id": paths.hub_id,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "library_class": "GeometryAwareFourierFusionLocalizer",
        "factory_method": "for_09_2_pooled",
        "variant_id": VARIANT_ID,
        "fusion_pattern": FUSION_PATTERN_09_2_POOLED,
        "backbone_kind": "pooled",
        "n_views": n_views,
        "view_angles_deg": view_angles_deg,
        "geometry_features": ["sin_theta", "cos_theta"],
        "geometry_mode": "concat",
        "fusion_hidden": 128,
        "fusion_depth": 1,
        "n_outputs": len(Y_FIELDS),
        "y_fields": list(Y_FIELDS),
        "x_field": freeze.x_field,
        "image_normalize": freeze.image_normalize,
        "input_channels": 1,
        "image_height": h,
        "image_width": w,
        "lr": float(metrics.get("lr", lr)),
        "n_params": int(metrics.get("n_params", n_params)),
        "source_checkpoint": f"checkpoints/m9/{M09_E2E_POOLED_GEOMETRY_FUSION}",
        "source_experiment_id": "m09_e2e_pooled_geometry_fusion",
        "source_state_key": STATE_KEY,
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

    return M9CompactExportResult(
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
    "ARCHITECTURE",
    "DEFAULT_STUDY_CHECKPOINT",
    "FOURIER_STUDY_CHECKPOINT",
    "MODEL_KEY",
    "M9CompactExportResult",
    "PROTOCOL",
    "STATE_KEY",
    "VARIANT_ID",
    "export_camera_orbit_compact_09_2b",
    "resolve_camera_orbit_compact_09_2b_paths",
]
