"""Export M10 Step 3 / 10_2 hierarchical pooled model into a local Hugging Face clone."""

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
    FUSION_PATTERN_10_2_POOLED,
    HierarchicalLightThenCameraFusionLocalizer,
)
from tomography_ml.studies.study_checkpoints import M10_HIERARCHICAL_LIGHT_THEN_CAMERA
from tomography_ml_huggingface.hf_local import (
    HfModelLocalPaths,
    resolve_hf_model_paths,
)
from tomography_ml_huggingface.hub_model_card import (
    DATASET_HUB_ID,
    FINAL_REPORT_URL,
    ModelCardEvalSpec,
    REPO_URL,
    WEIGHTS_NAME,
    build_evaluation_results_prose,
    build_model_card_frontmatter,
    build_results_table,
    write_hub_model_artifacts,
)

DEFAULT_STUDY_CHECKPOINT = Path("checkpoints/m10") / M10_HIERARCHICAL_LIGHT_THEN_CAMERA
MODEL_KEY = "gummybear_hierarchical_fusion"
PROTOCOL = "10_2"
ARCHITECTURE = "pooled_gap"
VARIANT_ID = "m10_2_hierarchical_pooled_light_then_camera"
DESCRIBE_VARIANT_ID = "m10_2_hierarchical_pooled_light_then_camera_fusion"
STATE_KEY = "model_pooled"
Y_FIELDS = ("particle_x", "particle_y", "particle_z")


@dataclass(frozen=True)
class M10HierarchicalExportResult:
    """Artefacts written for ``tbhugging/gummybear_hierarchical_fusion``."""

    hub_id: str
    local_clone: Path
    checkpoint_path: Path
    weights_path: Path
    config_path: Path
    readme_path: Path
    n_params: int
    lr: float
    metrics: dict[str, float]


def resolve_gummybear_hierarchical_fusion_paths(
    repo_root: Path,
    *,
    local_toml: Path | None = None,
) -> HfModelLocalPaths:
    """Read ``models.gummybear_hierarchical_fusion`` from gitignored local.toml."""
    return resolve_hf_model_paths(
        repo_root, model_key=MODEL_KEY, local_toml=local_toml
    )


def _metrics_from_m10_comparison(
    comparison: Any,
    *,
    variant_id: str = VARIANT_ID,
    selected_lrs: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Pull pooled hierarchical Euclidean RMSE (+ n_params / lr) from comparison."""
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
    if selected_lrs is not None and selected_lrs.get("pooled") is not None:
        out["lr"] = float(selected_lrs["pooled"])
    elif not sub.empty and "lr_stage_b" in sub.columns:
        out["lr"] = float(sub["lr_stage_b"].iloc[0])
    return out


def _format_angle_list(angles_deg: Sequence[float]) -> str:
    return ", ".join(f"{float(a):g}°" for a in angles_deg)


def _format_orbit_summary(angles_deg: Sequence[float]) -> str:
    angles = [float(a) for a in angles_deg]
    if len(angles) <= 8:
        return _format_angle_list(angles)
    stride = angles[1] - angles[0] if len(angles) > 1 else 0.0
    return (
        f"{len(angles)} views on {stride:g}° stride "
        f"({angles[0]:g}°–{angles[-1]:g}°)"
    )


def _assert_pooled_hierarchical_checkpoint(blob: Mapping[str, Any], ckpt: Path) -> None:
    """Require the pooled hierarchical head in the study checkpoint."""
    if STATE_KEY not in blob or blob[STATE_KEY] is None:
        raise KeyError(
            f"{display_path(ckpt)} missing {STATE_KEY!r} "
            "(run Final Report M10 Step 3 / 10_2 pooled hierarchical first)."
        )
    describe = blob.get("describe_pooled")
    if isinstance(describe, Mapping):
        kind = str(describe.get("backbone_kind") or "").strip().lower()
        if kind and kind != "pooled":
            raise ValueError(
                f"{display_path(ckpt)} describe_pooled backbone_kind={kind!r}; "
                f"expected pooled GAP export."
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
    n_lights = int(config["n_lights"])
    n_cameras = int(config["n_cameras"])
    light_angles = [float(a) for a in config["light_angles_deg"]]
    camera_angles = [float(a) for a in config["camera_angles_deg"]]
    y_fields = [str(y) for y in config["y_fields"]]
    y_bullets = "\n".join(f"- {y}" for y in y_fields)
    eval_spec = ModelCardEvalSpec(
        task_name="Hierarchical illumination × camera-orbit localisation (xyz)",
        dataset_name="GummyBear Tomography (M10 joint illumination × camera grid)",
        source_name=f"Final Report M10 Step 3 / {PROTOCOL} (pooled GAP)",
        tags=(
            "tomography",
            "particle-localization",
            "cnn",
            "pooled",
            "gap",
            "multiview",
            "hierarchical-fusion",
            "illumination",
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
        f"Given **{n_lights}** illumination settings × **{n_cameras}** camera-orbit views "
        f"of {h}×{w} projections from the GummyBear Tomography dataset, the model predicts",
        "",
        y_bullets,
        "",
        "coordinates of the embedded particle.",
        "",
        "",
        "## Architecture",
        "",
        "Per-view **GAP (pooled) CNN trunk** → **light sin/cos tokens** fused within each camera →",
        "**camera sin/cos tokens** fused across the orbit "
        f"(`{config['fusion_pattern']}`).",
        "",
        f"Final Report **M10 Step 3 / {PROTOCOL}** — pooled hierarchical head (`{STATE_KEY}`).",
        "",
        f"- Hub: [{hub_id}]({hub_url})",
        "- Companion dataset: "
        f"[{DATASET_HUB_ID}](https://huggingface.co/datasets/{DATASET_HUB_ID})",
        "- Runnable report: "
        f"[GummyBearTomography_Final_Report.ipynb]({FINAL_REPORT_URL}) "
        f"(M10 Step 3 — {PROTOCOL} pooled ladder)",
        "- Source study checkpoint: "
        f"[{ckpt_rel}]"
        f"(https://huggingface.co/datasets/{DATASET_HUB_ID}/blob/main/"
        f"{ckpt_rel}) (`{STATE_KEY}` / `{DESCRIBE_VARIANT_ID}`)",
        "",
        "## Training configuration",
        "",
        f"- Protocol: `{config['protocol']}` (pooled GAP hierarchical; excludes Fourier 10_2)",
        f"- Backbone: `{config['architecture']}` (`{config['backbone_kind']}` trunk)",
        f"- Input field: `{config['x_field']}`",
        f"- Normalisation: `{config['image_normalize']}`",
        f"- Illumination orbit: `{_format_angle_list(light_angles)}` ({n_lights} lights)",
        f"- Camera orbit: `{_format_orbit_summary(camera_angles)}` ({n_cameras} views)",
        f"- Flat layout: `{config['flat_layout']}`",
        f"- Geometry: `{', '.join(config['geometry_features'])}`",
        f"- Fusion: hidden `{config['fusion_hidden']}`, depth `{config['fusion_depth']}`, "
        f"camera latent `{config['camera_latent_dim']}`",
        f"- Targets: `{', '.join(y_fields)}`",
        f"- Builder: `{config['library_class']}.{config['factory_method']}()`",
        f"- Variant: `{config['variant_id']}`",
        f"- Trainable parameters: `{config['n_params']}`",
        f"- Stage-B learning rate: `{config['lr']}`",
        "",
        *build_evaluation_results_prose(
            eval_spec=eval_spec,
            protocol_lines=[
                f"- Protocol: Final Report **M10 Step 3 / {PROTOCOL}** "
                f"({n_lights} lights × {n_cameras} cameras, "
                f"`anomaly_ref`, `per_image_zscore`, **GAP pooled trunk only**, "
                "hierarchical light-then-camera fusion; excludes Fourier 10_2)",
            ],
            checkpoint_markdown=(
                "- Source checkpoint: "
                f"[{ckpt_rel}]"
                f"(https://huggingface.co/datasets/{DATASET_HUB_ID}/blob/main/"
                f"{ckpt_rel}) (`{STATE_KEY}` only)"
            ),
        ),
        "Scores match the Final Report M10 Step 3 **10_2 pooled GAP hierarchical** bar "
        "(not Fourier 10_2; not element-wise MSE).",
        "",
        *build_results_table(metrics),
        "Source: "
        f"[Final Report M10 Step 3 / {PROTOCOL}]({FINAL_REPORT_URL}).",
        "",
        "## Load",
        "",
        "```python",
        "import torch",
        f"# libraries from {REPO_URL}",
        "# Historical class name — use .for_10_2_pooled() for this GAP checkpoint only.",
        "from tomography_ml.localization.localize_multiview import (",
        "    HierarchicalLightThenCameraFusionLocalizer,",
        ")",
        "",
        f"n_lights = {n_lights}",
        f"n_cameras = {n_cameras}",
        f"light_angles_deg = {list(light_angles)}",
        f"camera_angles_deg = {list(camera_angles)}",
        "model = HierarchicalLightThenCameraFusionLocalizer.for_10_2_pooled(",
        "    n_cameras=n_cameras,",
        "    n_lights=n_lights,",
        "    camera_angles_deg=camera_angles_deg,",
        "    light_angles_deg=light_angles_deg,",
        "    flat_layout='light_major',",
        ")",
        f"views = torch.zeros(1, n_lights, n_cameras, 1, {h}, {w})",
        "model(views)  # materialise lazy layers",
        f"state = torch.load('{WEIGHTS_NAME}', map_location='cpu', weights_only=True)",
        "model.load_state_dict(state)",
        "model.eval()",
        "xyz = model(views)",
        "```",
        "",
        "Input tensor shape: `[batch, n_lights, n_cameras, channels, height, width]` "
        f"(flat `{config['flat_layout']}` layout also supported).",
        "Do **not** load with `.for_10_2()` (that builds the Fourier 10_2 trunk).",
        "",
        "## Inference",
        "",
        "For an example with worked download, model instanciation and inference, see: "
        "[11_3_test_gummybear_hierarchical_fusion.ipynb]"
        f"({REPO_URL}/blob/master/"
        "notebooks/milestone_11/11_3_test_gummybear_hierarchical_fusion.ipynb) "
        f"in the [{REPO_URL.split('//')[-1]}]({REPO_URL}) repository.",
        "",
    ]
    return "\n".join(lines)


def export_gummybear_hierarchical_fusion(
    repo_root: Path,
    *,
    local_toml: Path | None = None,
    checkpoint_path: Path | None = None,
    local_clone: Path | None = None,
    image_hw: tuple[int, int] = (128, 128),
) -> M10HierarchicalExportResult:
    """Extract M10 10_2 pooled hierarchical weights into the local Hub model clone."""
    repo_root = Path(repo_root).resolve()
    paths = resolve_gummybear_hierarchical_fusion_paths(
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
            "(run Final Report M10 Step 3 / 10_2 hierarchical pooled first)."
        )

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    _assert_pooled_hierarchical_checkpoint(blob, ckpt)
    state = blob[STATE_KEY]
    camera_angles_deg = [float(a) for a in blob.get("view_angles_deg") or ()]
    light_angles_deg = [float(a) for a in blob.get("light_angles_deg") or ()]
    if not camera_angles_deg or not light_angles_deg:
        raise KeyError(
            f"{display_path(ckpt)} missing view_angles_deg / light_angles_deg"
        )
    n_cameras = len(camera_angles_deg)
    n_lights = len(light_angles_deg)
    selected_lrs = blob.get("selected_lrs") or {}
    metrics = _metrics_from_m10_comparison(
        blob.get("comparison"),
        selected_lrs=selected_lrs if isinstance(selected_lrs, Mapping) else None,
    )
    describe = blob.get("describe_pooled") or {}

    freeze = m8_single_view_block_freeze()
    h, w = int(image_hw[0]), int(image_hw[1])
    model = HierarchicalLightThenCameraFusionLocalizer.for_10_2_pooled(
        n_cameras=n_cameras,
        n_lights=n_lights,
        camera_angles_deg=camera_angles_deg,
        light_angles_deg=light_angles_deg,
        flat_layout=str(describe.get("flat_layout") or "light_major"),
        fusion_hidden=int(describe.get("fusion_hidden") or 128),
        fusion_depth=int(describe.get("fusion_depth") or 1),
        camera_latent_dim=int(describe.get("camera_latent_dim") or 128),
    )
    dummy = torch.zeros(1, n_lights, n_cameras, 1, h, w)
    model.eval()
    with torch.no_grad():
        model(dummy)
    model.load_state_dict(state, strict=True)
    n_params = int(count_parameters(model))

    config: dict[str, Any] = {
        "model_type": MODEL_KEY,
        "hub_id": paths.hub_id,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "library_class": "HierarchicalLightThenCameraFusionLocalizer",
        "factory_method": "for_10_2_pooled",
        "variant_id": DESCRIBE_VARIANT_ID,
        "comparison_variant_id": VARIANT_ID,
        "fusion_pattern": FUSION_PATTERN_10_2_POOLED,
        "backbone_kind": "pooled",
        "n_lights": n_lights,
        "n_cameras": n_cameras,
        "light_angles_deg": light_angles_deg,
        "camera_angles_deg": camera_angles_deg,
        "angle_stride_deg": float(describe.get("angle_stride_deg") or 10.0),
        "flat_layout": str(describe.get("flat_layout") or "light_major"),
        "geometry_features": list(
            describe.get("geometry_features")
            or ["sin_light", "cos_light", "sin_camera", "cos_camera"]
        ),
        "fusion_hidden": int(describe.get("fusion_hidden") or 128),
        "fusion_depth": int(describe.get("fusion_depth") or 1),
        "camera_latent_dim": int(describe.get("camera_latent_dim") or 128),
        "n_outputs": len(Y_FIELDS),
        "y_fields": list(Y_FIELDS),
        "x_field": freeze.x_field,
        "image_normalize": freeze.image_normalize,
        "input_channels": 1,
        "image_height": h,
        "image_width": w,
        "lr": float(metrics.get("lr", selected_lrs.get("pooled", 3e-4))),
        "n_params": int(metrics.get("n_params", n_params)),
        "source_checkpoint": f"checkpoints/m10/{M10_HIERARCHICAL_LIGHT_THEN_CAMERA}",
        "source_experiment_id": "m10_2_hierarchical_light_then_camera",
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

    return M10HierarchicalExportResult(
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
    "DESCRIBE_VARIANT_ID",
    "MODEL_KEY",
    "M10HierarchicalExportResult",
    "PROTOCOL",
    "STATE_KEY",
    "VARIANT_ID",
    "export_gummybear_hierarchical_fusion",
    "resolve_gummybear_hierarchical_fusion_paths",
]
