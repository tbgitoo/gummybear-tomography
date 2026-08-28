"""Tests for M11 M9 09_2B compact pooled Hub model export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tomography_ml.localization.builders import materialize_lazy_modules
from tomography_ml.localization.localize_multiview import GeometryAwareFourierFusionLocalizer
from tomography_ml_huggingface.m9_compact_export import (
    STATE_KEY,
    VARIANT_ID,
    export_camera_orbit_compact_09_2b,
    resolve_camera_orbit_compact_09_2b_paths,
)


REPO = Path(__file__).resolve().parents[1]
STUDY_CKPT = REPO / "checkpoints" / "m9" / "m09_e2e_pooled_geometry_fusion.pt"


@pytest.mark.skipif(not STUDY_CKPT.is_file(), reason="M9 pooled e2e checkpoint absent")
def test_export_camera_orbit_compact_09_2b_to_tmp(tmp_path: Path) -> None:
    local_toml = tmp_path / "local.toml"
    clone = tmp_path / "camera_orbit_compact_09_2b"
    local_toml.write_text(
        "\n".join(
            [
                "[models.camera_orbit_compact_09_2b]",
                'hub_id = "tbhugging/camera_orbit_compact_09_2b"',
                'hub_url = "https://huggingface.co/tbhugging/camera_orbit_compact_09_2b"',
                f'local_clone = "{clone}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = export_camera_orbit_compact_09_2b(
        REPO,
        local_toml=local_toml,
        checkpoint_path=STUDY_CKPT,
    )
    assert result.weights_path.is_file()
    assert result.config_path.is_file()
    assert result.readme_path.is_file()
    cfg = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert cfg["protocol"] == "09_2B"
    assert cfg["architecture"] == "pooled_gap"
    assert cfg["backbone_kind"] == "pooled"
    assert cfg["variant_id"] == VARIANT_ID
    assert cfg["fusion_pattern"] == "e2e_pooled_geometry_fusion"
    assert cfg["n_views"] == 6
    assert cfg["source_state_key"] == STATE_KEY
    assert cfg["n_params"] == result.n_params
    assert "validation_MSE" not in result.metrics
    assert "validation_RMSE_total" in result.metrics
    assert "test_RMSE_total" in result.metrics
    readme = result.readme_path.read_text(encoding="utf-8")
    assert "## What this model does" in readme
    assert "## Architecture" in readme
    assert "## Training configuration" in readme
    assert "## Evaluation Results" in readme
    assert "09_2B" in readme
    assert "GAP" in readme or "global average pooling" in readme.lower()
    assert "not" in readme.lower() and "09_2A" in readme
    assert "pooled" in readme.lower()
    assert "pipeline_tag: image-feature-extraction" in readme
    assert "model-index:" in readme
    assert "tbhugging/gummybear-tomography" in readme
    assert "11_2_test_camera_orbit_compact_09_2b.ipynb" in readme
    assert "11_2_camera_orbit_compact_09_2b_export.ipynb" not in readme
    assert "This Hub repo publishes" not in readme
    assert "not the larger 09_3" not in readme

    n_views = cfg["n_views"]
    h, w = cfg["image_height"], cfg["image_width"]
    model = GeometryAwareFourierFusionLocalizer.for_09_2_pooled(
        n_views=n_views,
        view_angles_deg=cfg["view_angles_deg"],
    )
    materialize_lazy_modules(model, torch.zeros(1, n_views, 1, h, w))
    state = torch.load(result.weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    paths = resolve_camera_orbit_compact_09_2b_paths(REPO, local_toml=local_toml)
    assert paths.local_clone == clone.resolve()
    assert paths.hub_id == "tbhugging/camera_orbit_compact_09_2b"


FOURIER_CKPT = REPO / "checkpoints" / "m9" / "m09_e2e_fourier_geometry_fusion.pt"


@pytest.mark.skipif(not FOURIER_CKPT.is_file(), reason="M9 Fourier e2e checkpoint absent")
def test_export_rejects_fourier_checkpoint(tmp_path: Path) -> None:
    local_toml = tmp_path / "local.toml"
    clone = tmp_path / "camera_orbit_compact_09_2b"
    local_toml.write_text(
        "\n".join(
            [
                "[models.camera_orbit_compact_09_2b]",
                'hub_id = "tbhugging/camera_orbit_compact_09_2b"',
                'hub_url = "https://huggingface.co/tbhugging/camera_orbit_compact_09_2b"',
                f'local_clone = "{clone}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="09_2A Fourier|family='fourier'"):
        export_camera_orbit_compact_09_2b(
            REPO,
            local_toml=local_toml,
            checkpoint_path=FOURIER_CKPT,
        )
