"""Shared lightweight fixtures for artifact-only M6 validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def m6_sequence_factory(tmp_path: Path):
    """Create small, valid M6 sequence artifacts without running physics."""

    counter = 0

    def make_sequence(
        *,
        anomaly: bool = False,
        frame_count: int = 2,
        image_size: tuple[int, int] = (8, 8),
    ):
        nonlocal counter
        sequence_id = f"m6_validation_{counter:03d}"
        counter += 1
        sequence_dir = tmp_path / sequence_id
        role_names = ["clean", "particle", "observed"]
        if anomaly:
            role_names.append("anomaly")
        for role in role_names:
            (sequence_dir / role).mkdir(parents=True, exist_ok=True)

        frames = []
        height, width = image_size
        for frame_index in range(frame_count):
            angle = float(frame_index * 60)
            stem = (
                f"{sequence_id}_frame_{frame_index:04d}_angle_{angle:+08.2f}"
            )
            clean = np.full((height, width), 20 + frame_index, dtype=np.uint8)
            particle = clean.copy()
            particle[2:6, 2:6] = 80 + frame_index
            filenames = {}
            for role, pixels in (
                ("clean", clean),
                ("particle", particle),
                ("observed", particle),
            ):
                relative = f"{role}/{stem}.jpg"
                Image.fromarray(pixels, mode="L").save(
                    sequence_dir / relative,
                    format="JPEG",
                    quality=95,
                    subsampling=0,
                )
                filenames[role] = relative
                raw_relative = f"{role}/{stem}.raw.tif"
                Image.fromarray(pixels.astype(np.float32), mode="F").save(
                    sequence_dir / raw_relative,
                    format="TIFF",
                )
                filenames[f"{role}_raw"] = raw_relative
            if anomaly:
                anomaly_pixels = np.full((height, width), 128, dtype=np.uint8)
                relative = f"anomaly/{stem}.png"
                Image.fromarray(anomaly_pixels, mode="L").save(
                    sequence_dir / relative,
                    format="PNG",
                )
                filenames["anomaly"] = relative
            frames.append(
                {
                    "frame_index": frame_index,
                    "angle_deg": angle,
                    "axis": [0.0, 0.0, 1.0],
                    "camera_kind": "orbit",
                    "camera_position": [0.0, -80.0, 2.5],
                    "look_at": [0.0, 0.0, 2.5],
                    "up": [0.0, 0.0, 1.0],
                    "resolution": [height, width],
                    "fov_deg": 35.0,
                    "filenames": filenames,
                    "anomaly_preview_max_abs": 0.1 if anomaly else None,
                }
            )

        def setup(sheet: str) -> dict:
            return {
                "source_sheet": sheet,
                "source_excel_row": 2,
                "workbook_name": "m6_test.xlsx",
                "workbook_sheet": sheet,
            }

        manifest = {
            "schema_version": "1.2-m6-draft",
            "generator_version": "m6-draft",
            "sequence_id": sequence_id,
            "created_utc": "2026-07-19T09:00:00+00:00",
            "split": "train",
            "seed": 42,
            "forward_model_tier": "m5_refractive_diffusion_particle_perturbation",
            "representation": {
                "image_format": "jpg",
                "jpeg_quality": 95,
                "image_domain": "camera_intensity",
                "composition_domain": "linear_camera_intensity_before_jpeg",
                "composition_policy": "pre_jpeg_numeric_arrays",
                "anomaly_definition": "particle_minus_clean",
                "observed_definition": "particle_no_corruption",
                "pixel_orientation": {
                    "camera_up": "image_top",
                    "transform_from_camera_sample_grid": "flip_axis_0",
                },
                "anomaly_preview": {
                    "authoritative": False,
                    "format": "png" if anomaly else None,
                    "mapping": (
                        "signed_per_frame_zero_centered" if anomaly else None
                    ),
                },
            },
            "roles": {
                "clean": "clean",
                "particle": "particle",
                "observed": "observed",
                **({"anomaly_preview": "anomaly"} if anomaly else {}),
            },
            "phantom": {
                "phantom_id": "proto_bear",
                "stl_path": "cad/proto_bear.stl",
                "stl_sha256": "a" * 64,
            },
            "workbook": {
                "workbook_path": "configs/m6/m6_test.xlsx",
                "sha256": "b" * 64,
                "sequence_sheet": "sequences",
                "sequence_excel_row": 2,
            },
            "setups": {
                "optical": setup("optical_setups"),
                "particle": setup("particles"),
                "diffusion": {
                    **setup("diffusion_setups"),
                    "effective": {
                        "D": 0.8,
                        "mu_a": 0.1,
                    },
                },
                "camera": setup("camera_schedules"),
                "corruption": setup("corruptions"),
            },
            "caches": {
                "clean_optical_cache_id": "c" * 64,
                "particle_source_cache_id": "d" * 64,
                "persistent_cache_used": False,
                "diffusion_operator_cache": None,
            },
            "generation": {
                "package_version": "0.0.1.dev0",
                "max_workers": 1,
                "runtime_settings": {},
                "stage_seconds": {
                    "clean_source": 0.1,
                    "particle_source": 0.1,
                    "diffusion_solves": 0.1,
                    "camera_capture": 0.1,
                },
                "diagnostics": {
                    "n_source_rays": 16,
                    "n_refracted_rays": 12,
                    "n_source_segments": 12,
                    "n_affected_paths": 2,
                    "source_assignment": "attenuated_chord",
                    "clean_solve_residual": 1e-12,
                    "particle_solve_residual": 1e-12,
                },
            },
            "frames": frames,
            "validation": {
                "role_alignment": "checked_before_write",
                "anomaly_identity": "particle_minus_clean_pre_jpeg",
                "post_jpeg_identity_checked": False,
            },
        }

        def write_manifest() -> None:
            (sequence_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        write_manifest()
        return sequence_dir, manifest, write_manifest

    return make_sequence
