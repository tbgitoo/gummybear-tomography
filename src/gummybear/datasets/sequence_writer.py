"""Atomic writer for multi-role sequence directories and manifests."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gummybear.datasets.role_images import (
    RoleImageEncoding,
    build_shared_encoding,
    encode_anomaly_preview,
    encode_camera_intensity,
    jpg_relative_to_raw_tif,
    orient_camera_image_for_storage,
    role_image_relative_to_raw_tif,
    write_float_raw_tif,
    write_uint8_image,
)


@dataclass(frozen=True)
class WrittenFrame:
    """One frame's relative role filenames after encoding.

    Attributes:
        frame_index, angle_deg: View ordering and metadata.
        filenames: Map of role keys to repo-relative paths under the sequence dir.
        anomaly_preview_max_abs: Peak |particle-clean| when anomaly PNG is written.
    """
    frame_index: int
    angle_deg: float
    filenames: dict[str, str]
    anomaly_preview_max_abs: float | None


@dataclass(frozen=True)
class SequenceWriteResult:
    """Published sequence directory after atomic staging.

    Attributes:
        sequence_directory: Final sequence folder path.
        frames: Per-frame filename maps aligned with manifest ``frames``.
        encoding: Shared linear uint8 mapping used for display JPEG preview roles.
    """
    sequence_directory: Path
    frames: tuple[WrittenFrame, ...]
    encoding: RoleImageEncoding


def frame_filename(
    sequence_id: str,
    frame_index: int,
    angle_deg: float,
    *,
    extension: str,
    index_width: int = 4,
) -> str:
    """Return a lexically sortable multi-view filename.

    ``frame_<index>`` zero-padding is the ordering source of truth; ``angle`` is
    human metadata embedded in the name.

    Args:
        sequence_id: Sequence identifier prefix.
        frame_index: Zero-based view index.
        angle_deg: View angle in degrees (formatted with sign and width).
        extension: File extension without leading dot.
        index_width: Zero-pad width for ``frame_index``.

    Returns:
        Filename string ``{sequence_id}_frame_{index}_angle_{angle}.{ext}``.
    """
    angle_text = f"{float(angle_deg):+08.2f}"
    return (
        f"{sequence_id}_frame_{int(frame_index):0{index_width}d}"
        f"_angle_{angle_text}.{extension.lstrip('.').lower()}"
    )


def _publish_staged_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing sequence directory without an "
            f"explicit stale-regeneration policy: {destination}"
        )
    os.replace(staging, destination)


def write_sequence_roles(
    *,
    output_root: Path | str,
    sequence_id: str,
    angles_deg: Sequence[float],
    clean_frames: Sequence[np.ndarray],
    particle_frames: Sequence[np.ndarray],
    manifest: dict[str, Any],
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    write_anomaly_preview: bool = True,
) -> SequenceWriteResult:
    """Write aligned clean/particle/observed roles and ``manifest.json`` JSON file.

    ``observed`` equals ``particle`` when corruptions are absent. Each display JPEG preview role
    has a float32 ``.raw.tif`` sidecar with linear camera intensity. Optional
    ``anomaly`` PNG preview encodes signed ``particle - clean`` (not authoritative).
    The directory is staged and atomically published; existing owners are not
    overwritten without an explicit stale-removal policy.

    Args:
        output_root: Scenario directory containing ``sequence_id/`` folders.
        sequence_id: Sequence identifier and subdirectory name.
        angles_deg: Per-frame angles aligned with frame arrays.
        clean_frames, particle_frames: Linear ``[H, W]`` grayscale intensities.
        manifest: Pre-built manifest dict; ``frames`` and encoding fields updated.
        image_format: Must resolve to display JPEG preview (JPG) for Phase 2 datasets.
        jpeg_quality: Display JPEG preview quality 1–100 for display roles.
        write_anomaly_preview: When True, write PNG anomaly previews and float sidecars.

    Returns:
        SequenceWriteResult with final path, frame filenames, and encoding.
    """
    clean_frames = tuple(np.asarray(frame, dtype=float) for frame in clean_frames)
    particle_frames = tuple(np.asarray(frame, dtype=float) for frame in particle_frames)
    angles_deg = tuple(float(angle) for angle in angles_deg)
    count = len(clean_frames)
    if count == 0:
        raise ValueError("A sequence must contain at least one frame.")
    if len(particle_frames) != count or len(angles_deg) != count:
        raise ValueError("Angles, clean frames, and particle frames must align.")
    expected_shape = clean_frames[0].shape
    if len(expected_shape) != 2:
        raise ValueError("Role frames must be two-dimensional grayscale arrays.")
    if any(frame.shape != expected_shape for frame in clean_frames + particle_frames):
        raise ValueError("All role frames must have the same shape.")

    format_name = image_format.strip().lower()
    extension = "jpg" if format_name in {"jpg", "jpeg"} else format_name
    if extension != "jpg":
        raise ValueError("M6 Phase 2 dataset roles must use JPG.")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / sequence_id
    staging = root / f".{sequence_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False)
    for role in ("clean", "particle", "observed"):
        (staging / role).mkdir()
    if write_anomaly_preview:
        (staging / "anomaly").mkdir()

    encoding = build_shared_encoding(clean_frames, particle_frames)
    index_width = max(4, len(str(count - 1)))
    written: list[WrittenFrame] = []
    try:
        for frame_index, (angle, clean, particle) in enumerate(
            zip(angles_deg, clean_frames, particle_frames, strict=True)
        ):
            clean = orient_camera_image_for_storage(clean)
            particle = orient_camera_image_for_storage(particle)
            role_name = frame_filename(
                sequence_id,
                frame_index,
                angle,
                extension=extension,
                index_width=index_width,
            )
            raw_name = frame_filename(
                sequence_id,
                frame_index,
                angle,
                extension="raw.tif",
                index_width=index_width,
            )
            clean_pixels = encode_camera_intensity(clean, encoding)
            particle_pixels = encode_camera_intensity(particle, encoding)
            write_uint8_image(
                staging / "clean" / role_name,
                clean_pixels,
                image_format="jpg",
                jpeg_quality=jpeg_quality,
            )
            write_uint8_image(
                staging / "particle" / role_name,
                particle_pixels,
                image_format="jpg",
                jpeg_quality=jpeg_quality,
            )
            write_uint8_image(
                staging / "observed" / role_name,
                particle_pixels,
                image_format="jpg",
                jpeg_quality=jpeg_quality,
            )
            write_float_raw_tif(staging / "clean" / raw_name, clean)
            write_float_raw_tif(staging / "particle" / raw_name, particle)
            write_float_raw_tif(staging / "observed" / raw_name, particle)

            filenames = {
                "clean": f"clean/{role_name}",
                "particle": f"particle/{role_name}",
                "observed": f"observed/{role_name}",
            }
            # Companion float sidecars (same stem as JPG, extension .raw.tif).
            filenames["clean_raw"] = jpg_relative_to_raw_tif(filenames["clean"])
            filenames["particle_raw"] = jpg_relative_to_raw_tif(
                filenames["particle"]
            )
            filenames["observed_raw"] = jpg_relative_to_raw_tif(
                filenames["observed"]
            )
            anomaly_max_abs: float | None = None
            if write_anomaly_preview:
                anomaly_name = frame_filename(
                    sequence_id,
                    frame_index,
                    angle,
                    extension="png",
                    index_width=index_width,
                )
                anomaly_delta = np.asarray(particle, dtype=float) - np.asarray(
                    clean, dtype=float
                )
                anomaly_pixels, anomaly_max_abs = encode_anomaly_preview(
                    particle,
                    clean,
                )
                write_uint8_image(
                    staging / "anomaly" / anomaly_name,
                    anomaly_pixels,
                    image_format="png",
                )
                write_float_raw_tif(staging / "anomaly" / raw_name, anomaly_delta)
                filenames["anomaly"] = f"anomaly/{anomaly_name}"
                filenames["anomaly_raw"] = role_image_relative_to_raw_tif(
                    filenames["anomaly"]
                )

            written.append(
                WrittenFrame(
                    frame_index=frame_index,
                    angle_deg=angle,
                    filenames=filenames,
                    anomaly_preview_max_abs=anomaly_max_abs,
                )
            )

        manifest["frames"] = [
            {
                **manifest_frame,
                "filenames": written_frame.filenames,
                "anomaly_preview_max_abs": (written_frame.anomaly_preview_max_abs),
            }
            for manifest_frame, written_frame in zip(
                manifest["frames"],
                written,
                strict=True,
            )
        ]
        manifest["representation"]["encoding"] = {
            "mapping": encoding.mapping,
            "dtype": encoding.dtype,
            "intensity_min": encoding.intensity_min,
            "intensity_max": encoding.intensity_max,
        }
        raw_roles = ["clean", "particle", "observed"]
        if write_anomaly_preview:
            raw_roles.append("anomaly")
        manifest["representation"]["raw_float_sidecar"] = {
            "extension": "raw.tif",
            "dtype": "float32",
            "image_domain": "linear_camera_intensity",
            "roles": raw_roles,
            "anomaly_definition": "particle_minus_clean",
            "naming": "same_stem_as_display_image_with_raw_tif_extension",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_staged_directory(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return SequenceWriteResult(
        sequence_directory=destination,
        frames=tuple(written),
        encoding=encoding,
    )
