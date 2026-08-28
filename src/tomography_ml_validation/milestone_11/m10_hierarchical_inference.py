"""Download and run inference for ``tbhugging/gummybear_hierarchical_fusion``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gummybear.paths import display_path
from tomography_ml.localization.builders import count_parameters
from tomography_ml.localization.localize_multiview import (
    HierarchicalLightThenCameraFusionLocalizer,
)
from tomography_ml_validation.milestone_11.hub_download import (
    DEFAULT_HUB_DOWNLOAD_TIMEOUT_S,
    HubDownloadError,
    download_hub_model_snapshot,
)
from tomography_ml_validation.milestone_11.hub_model_card import CONFIG_NAME, WEIGHTS_NAME
from tomography_ml_validation.milestone_11.m10_hierarchical_export import (
    ARCHITECTURE,
    DESCRIBE_VARIANT_ID,
    MODEL_KEY,
    PROTOCOL,
    STATE_KEY,
)


DEFAULT_HUB_ID = "tbhugging/gummybear_hierarchical_fusion"


@dataclass(frozen=True)
class LoadedHubGummybearHierarchicalFusion:
    """In-memory Hub model + config for M10 10_2 pooled hierarchical localisation."""

    hub_id: str
    snapshot_dir: Path
    config: dict[str, Any]
    model: torch.nn.Module
    n_params: int


@dataclass(frozen=True)
class HierarchicalSmokeResult:
    """Contract-shaped forward pass on a zero tensor (Hub smoke test)."""

    y_pred: tuple[float, float, float]
    input_shape: tuple[int, ...]


def download_gummybear_hierarchical_fusion(
    *,
    hub_id: str = DEFAULT_HUB_ID,
    revision: str | None = None,
    local_dir: Path | str | None = None,
    timeout_s: float = DEFAULT_HUB_DOWNLOAD_TIMEOUT_S,
) -> Path:
    """Fetch the published 10_2 pooled hierarchical model from the Hub."""
    return download_hub_model_snapshot(
        hub_id,
        revision=revision,
        local_dir=local_dir,
        timeout_s=timeout_s,
    )


def load_gummybear_hierarchical_fusion(
    model_dir: Path | str,
    *,
    hub_id: str = DEFAULT_HUB_ID,
    device: torch.device | str = "cpu",
) -> LoadedHubGummybearHierarchicalFusion:
    """Load ``config.json`` + ``pytorch_model.bin`` from a Hub snapshot / clone."""
    model_dir = Path(model_dir)
    config_path = model_dir / CONFIG_NAME
    weights_path = model_dir / WEIGHTS_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {display_path(config_path)}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing {display_path(weights_path)}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("protocol") or "") != PROTOCOL:
        raise ValueError(
            f"Expected protocol={PROTOCOL!r}, got {config.get('protocol')!r}"
        )
    if str(config.get("architecture") or "") != ARCHITECTURE:
        raise ValueError(
            f"Expected architecture={ARCHITECTURE!r}, got {config.get('architecture')!r}"
        )
    if str(config.get("variant_id") or "") != DESCRIBE_VARIANT_ID:
        raise ValueError(
            f"Expected variant_id={DESCRIBE_VARIANT_ID!r}, "
            f"got {config.get('variant_id')!r}"
        )

    n_lights = int(config["n_lights"])
    n_cameras = int(config["n_cameras"])
    h = int(config.get("image_height") or 128)
    w = int(config.get("image_width") or 128)
    model = HierarchicalLightThenCameraFusionLocalizer.for_10_2_pooled(
        n_cameras=n_cameras,
        n_lights=n_lights,
        camera_angles_deg=[float(a) for a in config["camera_angles_deg"]],
        light_angles_deg=[float(a) for a in config["light_angles_deg"]],
        flat_layout=str(config.get("flat_layout") or "light_major"),
        fusion_hidden=int(config.get("fusion_hidden") or 128),
        fusion_depth=int(config.get("fusion_depth") or 1),
        camera_latent_dim=int(config.get("camera_latent_dim") or 128),
    )
    dummy = torch.zeros(1, n_lights, n_cameras, 1, h, w, device=device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        model(dummy)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return LoadedHubGummybearHierarchicalFusion(
        hub_id=str(config.get("hub_id") or hub_id),
        snapshot_dir=model_dir,
        config=config,
        model=model,
        n_params=int(count_parameters(model)),
    )


def load_gummybear_hierarchical_fusion_from_hub(
    *,
    hub_id: str = DEFAULT_HUB_ID,
    revision: str | None = None,
    local_dir: Path | str | None = None,
    timeout_s: float = DEFAULT_HUB_DOWNLOAD_TIMEOUT_S,
    device: torch.device | str = "cpu",
) -> LoadedHubGummybearHierarchicalFusion:
    """Download from the published Hub repo (no cache) and load the localiser."""
    snap = download_gummybear_hierarchical_fusion(
        hub_id=hub_id,
        revision=revision,
        local_dir=local_dir,
        timeout_s=timeout_s,
    )
    return load_gummybear_hierarchical_fusion(snap, hub_id=hub_id, device=device)


def run_hub_contract_smoke_inference(
    loaded: LoadedHubGummybearHierarchicalFusion,
    *,
    device: torch.device | str | None = None,
) -> HierarchicalSmokeResult:
    """Hub smoke test: zero tensor at the published `[I,V,C,H,W]` contract shape."""
    cfg = loaded.config
    n_lights = int(cfg["n_lights"])
    n_cameras = int(cfg["n_cameras"])
    h = int(cfg.get("image_height") or 128)
    w = int(cfg.get("image_width") or 128)
    dev = device if device is not None else next(loaded.model.parameters()).device
    views = torch.zeros(1, n_lights, n_cameras, 1, h, w, device=dev)
    with torch.no_grad():
        pred = loaded.model(views).detach().cpu().numpy()[0]
    return HierarchicalSmokeResult(
        y_pred=(float(pred[0]), float(pred[1]), float(pred[2])),
        input_shape=tuple(views.shape),
    )


__all__ = [
    "DEFAULT_HUB_DOWNLOAD_TIMEOUT_S",
    "DEFAULT_HUB_ID",
    "HierarchicalSmokeResult",
    "HubDownloadError",
    "LoadedHubGummybearHierarchicalFusion",
    "MODEL_KEY",
    "download_gummybear_hierarchical_fusion",
    "load_gummybear_hierarchical_fusion",
    "load_gummybear_hierarchical_fusion_from_hub",
    "run_hub_contract_smoke_inference",
]
