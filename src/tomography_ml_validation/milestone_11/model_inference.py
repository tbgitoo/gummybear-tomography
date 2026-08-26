"""Download and run inference for ``tbhugging/singleview_cnn_fourier``."""

from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from gummybear.paths import (
    display_path,
    display_safe_warnings,
    install_display_safe_warning_paths,
)
from tomography_ml.gummybear_data_catalog.task_dataset import (
    IMAGE_NORMALIZE_PER_IMAGE_ZSCORE,
    apply_image_normalize,
    load_role_array,
)
from tomography_ml.gummybear_data_catalog.catalog import RoleRef
from tomography_ml.localization.builders import (
    count_parameters,
    materialize_lazy_modules,
)
from tomography_ml.studies.single_view_m8 import make_m8_single_view_model
from tomography_ml_validation.milestone_11.model_export import (
    ARCH,
    CONFIG_NAME,
    MODEL_KEY,
    WEIGHTS_NAME,
)

DEFAULT_HUB_ID = "tbhugging/singleview_cnn_fourier"
DEFAULT_EXAMPLE_SEQUENCE = "bear_m8_high_000004"
DEFAULT_KEEP_ANGLE_DEG = 180.0
DEFAULT_HUB_DOWNLOAD_TIMEOUT_S = 30.0


class HubDownloadError(RuntimeError):
    """Published Hub snapshot could not be fetched (no local-cache fallback)."""


@dataclass(frozen=True)
class LoadedHubFourierLocalizer:
    """In-memory Hub model + config for single-view Fourier localisation."""

    hub_id: str
    snapshot_dir: Path
    config: dict[str, Any]
    model: torch.nn.Module
    n_params: int


@dataclass(frozen=True)
class ExampleInferenceSample:
    """One packaged demo view + ground-truth xyz for a smoke inference."""

    sequence_id: str
    image_path: Path
    manifest_path: Path
    angle_deg: float
    y_true: tuple[float, float, float]
    views_chw: np.ndarray  # [1, C, H, W] after contract normalisation


@dataclass(frozen=True)
class InferenceResult:
    """Predicted xyz (+ optional Euclidean error vs truth)."""

    y_pred: tuple[float, float, float]
    y_true: tuple[float, float, float] | None
    euclidean_error: float | None
    sample: ExampleInferenceSample


def download_singleview_cnn_fourier(
    *,
    hub_id: str = DEFAULT_HUB_ID,
    revision: str | None = None,
    local_dir: Path | str | None = None,
    timeout_s: float = DEFAULT_HUB_DOWNLOAD_TIMEOUT_S,
) -> Path:
    """Fetch the published Hub snapshot over the network (no cache fallback).

    Always talks to Hugging Face: ``force_download=True``, Hub cache unused,
    ``local_files_only`` forbidden. Writes into ``local_dir`` or a fresh temp
    directory. Offline mode, missing network, or a hung request raise
    :class:`HubDownloadError` after ``timeout_s`` seconds.
    """
    timeout_s = float(timeout_s)
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0; got {timeout_s}")

    # tqdm / huggingface_hub emit import-time warnings with absolute
    # site-packages paths; install the showwarning patch before importing.
    install_display_safe_warning_paths()

    with display_safe_warnings():
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.constants import is_offline_mode
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "huggingface_hub is required for Hub download. "
                'Install with: pip install ".[hf]" -c requirements.txt'
            ) from exc

        if is_offline_mode():
            raise HubDownloadError(
                f"Cannot download {hub_id}: Hugging Face offline mode is enabled "
                "(HF_HUB_OFFLINE). This helper tests the published remote repo and "
                "does not use a local cache."
            )

        dest = (
            Path(local_dir)
            if local_dir is not None
            else Path(tempfile.mkdtemp(prefix="gummybear_hub_"))
        )
        dest.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "repo_id": hub_id,
            "repo_type": "model",
            "local_dir": str(dest),
            "force_download": True,
            "local_files_only": False,
            "etag_timeout": min(timeout_s, 10.0),
        }
        if revision is not None:
            kwargs["revision"] = revision

        def _fetch() -> Path:
            with display_safe_warnings():
                return Path(snapshot_download(**kwargs))

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_fetch)
            try:
                return future.result(timeout=timeout_s)
            except TimeoutError:
                future.cancel()
                raise HubDownloadError(
                    f"Timed out after {timeout_s:g}s downloading {hub_id} from "
                    "Hugging Face. Check network connectivity; this helper does "
                    "not fall back to a local cache."
                ) from None
            except HubDownloadError:
                raise
            except Exception as exc:
                raise HubDownloadError(
                    f"Could not download {hub_id} from Hugging Face: {exc}. "
                    "This helper tests the published remote repo and does not "
                    "fall back to a local cache."
                ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def load_singleview_cnn_fourier(
    model_dir: Path | str,
    *,
    hub_id: str = DEFAULT_HUB_ID,
    device: torch.device | str = "cpu",
) -> LoadedHubFourierLocalizer:
    """Load ``config.json`` + ``pytorch_model.bin`` from a Hub snapshot / clone."""
    model_dir = Path(model_dir)
    config_path = model_dir / CONFIG_NAME
    weights_path = model_dir / WEIGHTS_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {display_path(config_path)}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing {display_path(weights_path)}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("architecture") or ARCH) != ARCH:
        raise ValueError(
            f"Expected architecture={ARCH!r}, got {config.get('architecture')!r}"
        )
    y_fields = [str(y) for y in (config.get("y_fields") or ("particle_x", "particle_y", "particle_z"))]
    h = int(config.get("image_height") or 128)
    w = int(config.get("image_width") or 128)

    model = make_m8_single_view_model(ARCH, n_outputs=len(y_fields), device=device)
    materialize_lazy_modules(
        model, torch.zeros(1, 1, h, w, device=device, dtype=torch.float32)
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return LoadedHubFourierLocalizer(
        hub_id=str(config.get("hub_id") or hub_id),
        snapshot_dir=model_dir,
        config=config,
        model=model,
        n_params=int(count_parameters(model)),
    )


def load_singleview_cnn_fourier_from_hub(
    *,
    hub_id: str = DEFAULT_HUB_ID,
    revision: str | None = None,
    local_dir: Path | str | None = None,
    timeout_s: float = DEFAULT_HUB_DOWNLOAD_TIMEOUT_S,
    device: torch.device | str = "cpu",
) -> LoadedHubFourierLocalizer:
    """Download from the published Hub repo (no cache) and load the localiser."""
    snap = download_singleview_cnn_fourier(
        hub_id=hub_id,
        revision=revision,
        local_dir=local_dir,
        timeout_s=timeout_s,
    )
    return load_singleview_cnn_fourier(snap, hub_id=hub_id, device=device)


def _packaged_m8_demo_sequence_dir(sequence_id: str) -> Path:
    """Resolve a packaged ``m8_demo`` sequence directory from package data."""
    import tomography_ml_validation as _pkg

    sequence_dir = (
        Path(_pkg.__file__).resolve().parent
        / "test_data"
        / "data"
        / "generated"
        / "m8_demo"
        / sequence_id
    )
    if not (sequence_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            f"Packaged demo sequence missing manifest: {display_path(sequence_dir)}"
        )
    return sequence_dir


def _particle_xyz_from_manifest(manifest: Mapping[str, Any]) -> tuple[float, float, float]:
    try:
        item = manifest["resolved_job"]["setups"]["particles"]["items"][0]
        return (
            float(item["center_x"]),
            float(item["center_y"]),
            float(item["center_z"]),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise KeyError(
            "manifest missing resolved_job.setups.particles.items[0].center_{x,y,z}"
        ) from exc


def load_packaged_m8_demo_example(
    *,
    sequence_id: str = DEFAULT_EXAMPLE_SEQUENCE,
    keep_angle_deg: float = DEFAULT_KEEP_ANGLE_DEG,
    image_normalize: str = IMAGE_NORMALIZE_PER_IMAGE_ZSCORE,
) -> ExampleInferenceSample:
    """Load anomaly_ref at ``keep_angle_deg`` from the packaged M8 demo corpus."""
    sequence_dir = _packaged_m8_demo_sequence_dir(sequence_id)
    manifest_path = sequence_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    y_true = _particle_xyz_from_manifest(manifest)

    role = RoleRef(manifest_path=manifest_path, role_name="anomaly")
    views = load_role_array(role, keep_angles_deg=float(keep_angle_deg))
    views = apply_image_normalize(views, image_normalize)  # type: ignore[arg-type]
    # Prefer the raw.tif path for display / provenance.
    matches = sorted(
        sequence_dir.glob(f"anomaly/*angle_+{float(keep_angle_deg):07.2f}.raw.tif")
    )
    if not matches:
        matches = sorted(sequence_dir.glob("anomaly/*180*.raw.tif"))
    if not matches:
        raise FileNotFoundError(
            f"No anomaly raw.tif at {keep_angle_deg}° under "
            f"{display_path(sequence_dir / 'anomaly')}"
        )
    image_path = matches[0]

    return ExampleInferenceSample(
        sequence_id=sequence_id,
        image_path=image_path,
        manifest_path=manifest_path,
        angle_deg=float(keep_angle_deg),
        y_true=y_true,
        views_chw=np.asarray(views, dtype=np.float32),
    )


def predict_xyz(
    loaded: LoadedHubFourierLocalizer,
    views_vchw: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> tuple[float, float, float]:
    """Run forward pass; ``views`` is ``[V,C,H,W]`` or ``[B,C,H,W]`` (V or B = 1)."""
    arr = np.asarray(views_vchw, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        batch = torch.as_tensor(arr, dtype=torch.float32)  # [1,C,H,W]
    elif arr.ndim == 3:
        batch = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError(
            f"Expected [1,C,H,W] or [C,H,W] float views; got shape {arr.shape}"
        )
    dev = device if device is not None else next(loaded.model.parameters()).device
    batch = batch.to(dev)
    with torch.no_grad():
        pred = loaded.model(batch).detach().cpu().numpy()[0]
    return float(pred[0]), float(pred[1]), float(pred[2])


def run_packaged_demo_inference(
    loaded: LoadedHubFourierLocalizer,
    *,
    sequence_id: str = DEFAULT_EXAMPLE_SEQUENCE,
    keep_angle_deg: float = DEFAULT_KEEP_ANGLE_DEG,
) -> InferenceResult:
    """Download-ready smoke test: Hub weights × packaged demo anomaly @ 180°."""
    sample = load_packaged_m8_demo_example(
        sequence_id=sequence_id,
        keep_angle_deg=keep_angle_deg,
        image_normalize=str(
            loaded.config.get("image_normalize") or IMAGE_NORMALIZE_PER_IMAGE_ZSCORE
        ),
    )
    y_pred = predict_xyz(loaded, sample.views_chw)
    err = float(
        np.linalg.norm(np.asarray(y_pred) - np.asarray(sample.y_true), ord=2)
    )
    return InferenceResult(
        y_pred=y_pred,
        y_true=sample.y_true,
        euclidean_error=err,
        sample=sample,
    )


__all__ = [
    "DEFAULT_EXAMPLE_SEQUENCE",
    "DEFAULT_HUB_DOWNLOAD_TIMEOUT_S",
    "DEFAULT_HUB_ID",
    "DEFAULT_KEEP_ANGLE_DEG",
    "ExampleInferenceSample",
    "HubDownloadError",
    "InferenceResult",
    "LoadedHubFourierLocalizer",
    "MODEL_KEY",
    "download_singleview_cnn_fourier",
    "load_packaged_m8_demo_example",
    "load_singleview_cnn_fourier",
    "load_singleview_cnn_fourier_from_hub",
    "predict_xyz",
    "run_packaged_demo_inference",
]
