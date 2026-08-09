"""M6 role-image conversion and writing.

Clean and particle display JPEG previews (lossy uint8) share one linear exposure;
float32 ``.raw.tif`` sidecars hold authoritative linear camera intensity for catalog
loaders. Anomaly PNG is a display-only signed preview of ``particle - clean``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RoleImageEncoding:
    """Shared linear mapping from camera intensity to uint8 display JPEG preview roles.

    Clean and particle frames use one min/max span so ``particle - clean`` stays
    meaningful in float ``.raw.tif`` sidecars and anomaly previews.

    Attributes:
        intensity_min, intensity_max: Linear clip range across all role frames.
        dtype: Stored pixel dtype (``uint8`` for display JPEG preview roles).
        mapping: Mapping label (``shared_linear_clip``).
    """

    intensity_min: float
    intensity_max: float
    dtype: str = "uint8"
    mapping: str = "shared_linear_clip"


def orient_camera_image_for_storage(image: np.ndarray) -> np.ndarray:
    """Put camera-up (normally +z) at the top of the stored image.

    Camera ray grids enumerate their vertical coordinate from negative to
    positive camera-up. Image files and ``imshow`` enumerate rows from top to
    bottom, so axis 0 must be reversed when crossing that representation
    boundary.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Camera images must be two-dimensional.")
    return np.flip(image, axis=0)


def build_shared_encoding(
    clean_frames: Sequence[np.ndarray],
    particle_frames: Sequence[np.ndarray],
) -> RoleImageEncoding:
    """Choose one deterministic linear mapping for aligned role frames."""
    arrays = [
        np.asarray(frame, dtype=float)
        for frame in (*tuple(clean_frames), *tuple(particle_frames))
    ]
    if not arrays:
        raise ValueError("At least one clean or particle frame is required.")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Role frames must contain only finite values.")

    intensity_min = min(0.0, min(float(np.min(array)) for array in arrays))
    intensity_max = max(float(np.max(array)) for array in arrays)
    if intensity_max <= intensity_min:
        intensity_max = intensity_min + 1.0
    return RoleImageEncoding(
        intensity_min=float(intensity_min),
        intensity_max=float(intensity_max),
    )


def encode_camera_intensity(
    image: np.ndarray,
    encoding: RoleImageEncoding,
) -> np.ndarray:
    """Encode a linear camera-intensity image as uint8."""
    image = np.asarray(image, dtype=float)
    scale = encoding.intensity_max - encoding.intensity_min
    normalized = (image - encoding.intensity_min) / scale
    return np.rint(255.0 * np.clip(normalized, 0.0, 1.0)).astype(np.uint8)


def encode_anomaly_preview(
    particle: np.ndarray,
    clean: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Encode signed ``particle - clean`` around neutral gray."""
    delta = np.asarray(particle, dtype=float) - np.asarray(clean, dtype=float)
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_abs <= np.finfo(float).eps:
        return np.full(delta.shape, 128, dtype=np.uint8), 0.0
    encoded = 127.0 * np.clip(delta / max_abs, -1.0, 1.0) + 128.0
    return np.rint(encoded).astype(np.uint8), max_abs


def write_uint8_image(
    path: Path | str,
    pixels: np.ndarray,
    *,
    image_format: str,
    jpeg_quality: int = 95,
) -> Path:
    """Write one grayscale uint8 role image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="L")
    format_name = image_format.strip().lower()
    if format_name in {"jpg", "jpeg"}:
        image.save(
            path,
            format="JPEG",
            quality=int(jpeg_quality),
            subsampling=0,
            optimize=False,
        )
    elif format_name == "png":
        image.save(path, format="PNG")
    else:
        raise ValueError(f"Unsupported image format: {image_format!r}")
    return path


RAW_FLOAT_EXTENSION = "raw.tif"
_RAW_SOURCE_SUFFIXES = (".jpg", ".jpeg", ".png")


def role_image_relative_to_raw_tif(relative: str) -> str:
    """Map ``role/stem.jpg|.jpeg|.png`` to companion float ``role/stem.raw.tif`` sidecar."""
    lower = relative.lower()
    for suffix in _RAW_SOURCE_SUFFIXES:
        if lower.endswith(suffix):
            return relative[: -len(suffix)] + f".{RAW_FLOAT_EXTENSION}"
    raise ValueError(
        "Raw float sidecars are defined for display JPEG preview/PNG role images, "
        f"got {relative!r}."
    )


def jpg_relative_to_raw_tif(jpg_relative: str) -> str:
    """Map ``role/stem.jpg`` to the companion float ``.raw.tif`` sidecar ``role/stem.raw.tif``."""
    if not jpg_relative.lower().endswith(".jpg"):
        raise ValueError(
            "Raw float sidecars are defined only for display JPEG preview role images, "
            f"got {jpg_relative!r}."
        )
    return role_image_relative_to_raw_tif(jpg_relative)


def write_float_raw_tif(path: Path | str, values: np.ndarray) -> Path:
    """Write one grayscale float32 TIFF image file of linear camera intensity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Raw float role images must be two-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Raw float role images must contain only finite values.")
    Image.fromarray(array, mode="F").save(path, format="TIFF")
    return path
