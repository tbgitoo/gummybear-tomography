"""Tests for Hub M10 hierarchical pooled model inference helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tomography_ml_huggingface.m10_hierarchical_export import (
    export_gummybear_hierarchical_fusion,
)
from tomography_ml_huggingface.m10_hierarchical_inference import (
    load_gummybear_hierarchical_fusion,
    run_hub_contract_smoke_inference,
)


REPO = Path(__file__).resolve().parents[1]
STUDY_CKPT = REPO / "checkpoints" / "m10" / "m10_hierarchical_light_then_camera.pt"


@pytest.mark.skipif(
    not STUDY_CKPT.is_file(), reason="M10 hierarchical study checkpoint absent"
)
def test_hub_contract_smoke_from_exported_weights(tmp_path: Path) -> None:
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
    export_gummybear_hierarchical_fusion(
        REPO, local_toml=local_toml, checkpoint_path=STUDY_CKPT
    )
    loaded = load_gummybear_hierarchical_fusion(clone)
    result = run_hub_contract_smoke_inference(loaded)
    assert len(result.y_pred) == 3
    assert result.input_shape == (1, 6, 36, 1, 128, 128)
