"""Tests for M11 M10 10_2 hierarchical pooled Hub model export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tomography_ml.localization.localize_multiview import (
    HierarchicalLightThenCameraFusionLocalizer,
)
from tomography_ml_huggingface.m10_hierarchical_export import (
    DESCRIBE_VARIANT_ID,
    STATE_KEY,
    VARIANT_ID,
    export_gummybear_hierarchical_fusion,
    resolve_gummybear_hierarchical_fusion_paths,
)


REPO = Path(__file__).resolve().parents[1]
STUDY_CKPT = REPO / "checkpoints" / "m10" / "m10_hierarchical_light_then_camera.pt"


@pytest.mark.skipif(
    not STUDY_CKPT.is_file(), reason="M10 hierarchical study checkpoint absent"
)
def test_export_gummybear_hierarchical_fusion_to_tmp(tmp_path: Path) -> None:
    local_toml = tmp_path / "local.toml"
    clone = tmp_path / "gummybear_hierarchical_fusion"
    local_toml.write_text(
        "\n".join(
            [
                "[models.gummybear_hierarchical_fusion]",
                'hub_id = "tbhugging/gummybear_hierarchical_fusion"',
                'hub_url = "https://huggingface.co/tbhugging/gummybear_hierarchical_fusion"',
                f'local_clone = "{clone}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = export_gummybear_hierarchical_fusion(
        REPO,
        local_toml=local_toml,
        checkpoint_path=STUDY_CKPT,
    )
    assert result.weights_path.is_file()
    assert result.config_path.is_file()
    assert result.readme_path.is_file()
    cfg = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert cfg["protocol"] == "10_2"
    assert cfg["architecture"] == "pooled_gap"
    assert cfg["backbone_kind"] == "pooled"
    assert cfg["variant_id"] == DESCRIBE_VARIANT_ID
    assert cfg["comparison_variant_id"] == VARIANT_ID
    assert cfg["n_lights"] == 6
    assert cfg["n_cameras"] == 36
    assert cfg["source_state_key"] == STATE_KEY
    assert cfg["n_params"] == result.n_params
    assert "validation_RMSE_total" in result.metrics
    assert "test_RMSE_total" in result.metrics
    readme = result.readme_path.read_text(encoding="utf-8")
    assert "## What this model does" in readme
    assert "## Architecture" in readme
    assert "hierarchical" in readme.lower()
    assert "GAP" in readme or "pooled" in readme.lower()
    assert "11_3_test_gummybear_hierarchical_fusion.ipynb" in readme
    assert "11_3_gummybear_hierarchical_fusion_export.ipynb" not in readme
    assert "pipeline_tag: image-feature-extraction" in readme
    assert "model-index:" in readme
    assert "checkpoints/m10/m10_hierarchical_light_then_camera.pt" in readme

    model = HierarchicalLightThenCameraFusionLocalizer.for_10_2_pooled(
        n_cameras=cfg["n_cameras"],
        n_lights=cfg["n_lights"],
        camera_angles_deg=cfg["camera_angles_deg"],
        light_angles_deg=cfg["light_angles_deg"],
        flat_layout=cfg["flat_layout"],
    )
    dummy = torch.zeros(1, cfg["n_lights"], cfg["n_cameras"], 1, 128, 128)
    model(dummy)
    state = torch.load(result.weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    paths = resolve_gummybear_hierarchical_fusion_paths(REPO, local_toml=local_toml)
    assert paths.local_clone == clone.resolve()
    assert paths.hub_id == "tbhugging/gummybear_hierarchical_fusion"
