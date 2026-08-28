"""Download and run inference for ``tbhugging/camera_orbit_compact_09_2b``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gummybear.paths import display_path
from tomography_ml.gummybear_data_catalog.catalog import RoleRef
from tomography_ml.gummybear_data_catalog.task_dataset import (
    IMAGE_NORMALIZE_PER_IMAGE_ZSCORE,
    apply_image_normalize,
    load_role_array,
)
from tomography_ml.localization.builders import count_parameters, materialize_lazy_modules
from tomography_ml.localization.localize_multiview import GeometryAwareFourierFusionLocalizer
from tomography_ml_huggingface.hub_download import (
    DEFAULT_HUB_DOWNLOAD_TIMEOUT_S,
    HubDownloadError,
    download_hub_model_snapshot,
)
from tomography_ml_huggingface.hub_model_card import CONFIG_NAME, WEIGHTS_NAME
from tomography_ml_huggingface.m9_compact_export import (
    ARCHITECTURE,
    MODEL_KEY,
    PROTOCOL,
    VARIANT_ID,
)
from tomography_ml_huggingface.model_inference import (
    ExampleInferenceSample,
    InferenceResult,
    _packaged_m8_demo_sequence_dir,
    _particle_xyz_from_manifest,
)

DEFAULT_HUB_ID = "tbhugging/camera_orbit_compact_09_2b"
DEFAULT_EXAMPLE_SEQUENCE = "bear_m8_high_000004"
DEFAULT_VIEW_ANGLES_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
# Packaged m8_demo uses 36° stride; widen matching for smoke inference only.
DEMO_ANGLE_ATOL_DEG = 18.0


@dataclass(frozen=True)
class LoadedHubCameraOrbitCompact09_2b:
    """In-memory Hub model + config for M9 09_2B pooled camera-orbit localisation."""

    hub_id: str
    snapshot_dir: Path
    config: dict[str, Any]
    model: torch.nn.Module
    n_params: int


@dataclass(frozen=True)
class MultiviewExampleInferenceSample:
    """Packaged demo views stacked for M9 camera-orbit inference."""

    sequence_id: str
    manifest_path: Path
    view_angles_deg: tuple[float, ...]
    matched_angles_deg: tuple[float, ...]
    image_paths: tuple[Path, ...]
    y_true: tuple[float, float, float]
    views_vchw: np.ndarray  # [V, C, H, W] after contract normalisation


def download_camera_orbit_compact_09_2b(
    *,
    hub_id: str = DEFAULT_HUB_ID,
    revision: str | None = None,
    local_dir: Path | str | None = None,
    timeout_s: float = DEFAULT_HUB_DOWNLOAD_TIMEOUT_S,
) -> Path:
    """Fetch the published 09_2B GAP model from the Hub (no cache fallback)."""
    return download_hub_model_snapshot(
        hub_id,
        revision=revision,
        local_dir=local_dir,
        timeout_s=timeout_s,
    )


def load_camera_orbit_compact_09_2b(
    model_dir: Path | str,
    *,
    hub_id: str = DEFAULT_HUB_ID,
    device: torch.device | str = "cpu",
) -> LoadedHubCameraOrbitCompact09_2b:
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
    if str(config.get("variant_id") or "") != VARIANT_ID:
        raise ValueError(
            f"Expected variant_id={VARIANT_ID!r}, got {config.get('variant_id')!r}"
        )
    view_angles_deg = [float(a) for a in config["view_angles_deg"]]
    n_views = int(config["n_views"])
    if len(view_angles_deg) != n_views:
        raise ValueError(
            f"config n_views={n_views} but len(view_angles_deg)={len(view_angles_deg)}"
        )
    h = int(config.get("image_height") or 128)
    w = int(config.get("image_width") or 128)

    model = GeometryAwareFourierFusionLocalizer.for_09_2_pooled(
        n_views=n_views,
        view_angles_deg=view_angles_deg,
    )
    materialize_lazy_modules(
        model,
        torch.zeros(1, n_views, 1, h, w, device=device, dtype=torch.float32),
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return LoadedHubCameraOrbitCompact09_2b(
        hub_id=str(config.get("hub_id") or hub_id),
        snapshot_dir=model_dir,
        config=config,
        model=model,
        n_params=int(count_parameters(model)),
    )


def load_camera_orbit_compact_09_2b_from_hub(
    *,
    hub_id: str = DEFAULT_HUB_ID,
    revision: str | None = None,
    local_dir: Path | str | None = None,
    timeout_s: float = DEFAULT_HUB_DOWNLOAD_TIMEOUT_S,
    device: torch.device | str = "cpu",
) -> LoadedHubCameraOrbitCompact09_2b:
    """Download from the published Hub repo (no cache) and load the localiser."""
    snap = download_camera_orbit_compact_09_2b(
        hub_id=hub_id,
        revision=revision,
        local_dir=local_dir,
        timeout_s=timeout_s,
    )
    return load_camera_orbit_compact_09_2b(snap, hub_id=hub_id, device=device)


def _demo_matched_angles(
    manifest: Mapping[str, Any],
    view_angles_deg: Sequence[float],
    *,
    angle_atol_deg: float = DEMO_ANGLE_ATOL_DEG,
) -> tuple[float, ...]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        return tuple(float(a) for a in view_angles_deg)
    indexed = [
        float(frame["angle_deg"])
        for frame in frames
        if isinstance(frame, dict) and "angle_deg" in frame
    ]
    matched: list[float] = []
    for target in view_angles_deg:
        nearest = min(indexed, key=lambda angle: abs(angle - float(target)))
        if abs(nearest - float(target)) > float(angle_atol_deg):
            raise ValueError(
                f"No packaged demo frame within {angle_atol_deg}° of {target}°"
            )
        matched.append(nearest)
    return tuple(matched)


def load_packaged_m9_demo_multiview_example(
    *,
    sequence_id: str = DEFAULT_EXAMPLE_SEQUENCE,
    view_angles_deg: Sequence[float] = DEFAULT_VIEW_ANGLES_DEG,
    image_normalize: str = IMAGE_NORMALIZE_PER_IMAGE_ZSCORE,
    angle_atol_deg: float = DEMO_ANGLE_ATOL_DEG,
) -> MultiviewExampleInferenceSample:
    """Load nearest packaged M8-demo views for each M9 camera-orbit angle.

    The packaged ``m8_demo`` corpus uses a 36° stride, so frames are
    nearest-neighbour matches to the trained 60° orbit (illustrative only).
    """
    sequence_dir = _packaged_m8_demo_sequence_dir(sequence_id)
    manifest_path = sequence_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    y_true = _particle_xyz_from_manifest(manifest)
    angles = tuple(float(a) for a in view_angles_deg)
    matched = _demo_matched_angles(
        manifest, angles, angle_atol_deg=angle_atol_deg
    )

    role = RoleRef(manifest_path=manifest_path, role_name="anomaly")
    views = load_role_array(
        role,
        keep_angles_deg=angles,
        angle_atol_deg=angle_atol_deg,
    )
    views = apply_image_normalize(views, image_normalize)  # type: ignore[arg-type]

    image_paths: list[Path] = []
    for target in angles:
        matches = sorted(
            sequence_dir.glob(f"anomaly/*angle_+{float(target):07.2f}.raw.tif")
        )
        if not matches:
            # Fall back to nearest matched angle filename pattern.
            nearest = min(
                matched,
                key=lambda angle: abs(angle - float(target)),
            )
            matches = sorted(
                sequence_dir.glob(f"anomaly/*angle_+{nearest:07.2f}.raw.tif")
            )
        if not matches:
            raise FileNotFoundError(
                f"No anomaly raw.tif near {target}° under "
                f"{display_path(sequence_dir / 'anomaly')}"
            )
        image_paths.append(matches[0])

    return MultiviewExampleInferenceSample(
        sequence_id=sequence_id,
        manifest_path=manifest_path,
        view_angles_deg=angles,
        matched_angles_deg=matched,
        image_paths=tuple(image_paths),
        y_true=y_true,
        views_vchw=np.asarray(views, dtype=np.float32),
    )


def predict_multiview_xyz(
    loaded: LoadedHubCameraOrbitCompact09_2b,
    views_vchw: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, float, float]:
    """Run forward pass; ``views`` is ``[V,C,H,W]`` (stacked multi-view input)."""
    arr = np.asarray(views_vchw, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected [V,C,H,W] float views; got shape {arr.shape}")
    batch = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1,V,C,H,W]
    dev = device if device is not None else next(loaded.model.parameters()).device
    batch = batch.to(dev)
    with torch.no_grad():
        pred = loaded.model(batch).detach().cpu().numpy()[0]
    return float(pred[0]), float(pred[1]), float(pred[2])


def run_packaged_m9_demo_inference(
    loaded: LoadedHubCameraOrbitCompact09_2b,
    *,
    sequence_id: str = DEFAULT_EXAMPLE_SEQUENCE,
) -> InferenceResult:
    """Hub smoke test: 09_2B weights × nearest-neighbour packaged demo views."""
    view_angles = tuple(float(a) for a in loaded.config.get("view_angles_deg") or ())
    sample_mv = load_packaged_m9_demo_multiview_example(
        sequence_id=sequence_id,
        view_angles_deg=view_angles,
        image_normalize=str(
            loaded.config.get("image_normalize") or IMAGE_NORMALIZE_PER_IMAGE_ZSCORE
        ),
    )
    y_pred = predict_multiview_xyz(loaded, sample_mv.views_vchw)
    err = float(
        np.linalg.norm(np.asarray(y_pred) - np.asarray(sample_mv.y_true), ord=2)
    )
    # Reuse ExampleInferenceSample for the first orbit slot (display convenience).
    sample = ExampleInferenceSample(
        sequence_id=sample_mv.sequence_id,
        image_path=sample_mv.image_paths[0],
        manifest_path=sample_mv.manifest_path,
        angle_deg=float(sample_mv.view_angles_deg[0]),
        y_true=sample_mv.y_true,
        views_chw=sample_mv.views_vchw[:1],
    )
    return InferenceResult(
        y_pred=y_pred,
        y_true=sample_mv.y_true,
        euclidean_error=err,
        sample=sample,
    )


__all__ = [
    "DEFAULT_EXAMPLE_SEQUENCE",
    "DEFAULT_HUB_DOWNLOAD_TIMEOUT_S",
    "DEFAULT_HUB_ID",
    "DEFAULT_VIEW_ANGLES_DEG",
    "HubDownloadError",
    "InferenceResult",
    "LoadedHubCameraOrbitCompact09_2b",
    "MODEL_KEY",
    "MultiviewExampleInferenceSample",
    "download_camera_orbit_compact_09_2b",
    "load_camera_orbit_compact_09_2b",
    "load_camera_orbit_compact_09_2b_from_hub",
    "load_packaged_m9_demo_multiview_example",
    "predict_multiview_xyz",
    "run_packaged_m9_demo_inference",
]
