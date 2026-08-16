"""Per-view z-score anomaly PNGs for POV-Ray camera back-plates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .load_sample import PhysicalSetup

_STD_EPS = 1e-8
DEFAULT_ZSCORE_CLIP = 2.0


def per_image_zscore(image: np.ndarray) -> np.ndarray:
    """``(x - mean) / std`` on one 2-D view; constant views become zeros."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"anomaly image must be 2-D, got shape {arr.shape}")
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= _STD_EPS:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32, copy=False)


def zscore_to_gray_uint8(
    z: np.ndarray, *, clip: float = DEFAULT_ZSCORE_CLIP
) -> np.ndarray:
    """Greyscale ``cmap='gray'`` with symmetric clip at ``±clip`` σ.

    Mapping is ``z=0 → mid-grey``. Saturating |z| at ``clip`` keeps the bear
    silhouette visible instead of stretching to a single hot pixel.
    """
    limit = float(clip)
    if limit <= 0.0:
        raise ValueError(f"z-score clip must be > 0, got {limit}")
    arr = np.clip(np.asarray(z, dtype=float), -limit, limit)
    t = (arr + limit) / (2.0 * limit)
    g = np.clip(t * 255.0, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def write_zscore_png(
    raw_tif: Path,
    dest_png: Path,
    *,
    clip: float = DEFAULT_ZSCORE_CLIP,
) -> Path:
    """Load float ``.raw.tif``, z-score, write an 8-bit greyscale PNG."""
    with Image.open(raw_tif) as im:
        arr = np.asarray(im, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    rgb = zscore_to_gray_uint8(per_image_zscore(arr), clip=clip)
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(dest_png, format="PNG")
    return dest_png


def write_anomaly_zscore_plates(
    setup: PhysicalSetup,
    dest_dir: Path,
    *,
    stem: str,
    clip: float = DEFAULT_ZSCORE_CLIP,
) -> dict[float, str]:
    """Write one z-score PNG per anomaly frame. Values are POV basenames."""
    plates: dict[float, str] = {}
    for angle_deg, raw_path in setup.frame_anomaly_raw:
        name = f"{stem}_zscore_{float(angle_deg):.2f}deg.png"
        write_zscore_png(raw_path, dest_dir / name, clip=clip)
        plates[float(angle_deg)] = name
    return plates
