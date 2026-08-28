"""Tests for Hub M9 09_2B pooled model inference helpers (offline via local export)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tomography_ml_validation.milestone_11.m9_compact_export import (
    export_camera_orbit_compact_09_2b,
)
from tomography_ml_validation.milestone_11.m9_compact_inference import (
    load_camera_orbit_compact_09_2b,
    load_packaged_m9_demo_multiview_example,
    predict_multiview_xyz,
    run_packaged_m9_demo_inference,
)


REPO = Path(__file__).resolve().parents[1]
STUDY_CKPT = REPO / "checkpoints" / "m9" / "m09_e2e_pooled_geometry_fusion.pt"


@pytest.mark.skipif(not STUDY_CKPT.is_file(), reason="M9 pooled e2e checkpoint absent")
def test_inference_on_packaged_demo_from_exported_weights(tmp_path: Path) -> None:
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
    export_camera_orbit_compact_09_2b(
        REPO, local_toml=local_toml, checkpoint_path=STUDY_CKPT
    )
    loaded = load_camera_orbit_compact_09_2b(clone)
    sample = load_packaged_m9_demo_multiview_example()
    assert sample.views_vchw.shape[0] == loaded.config["n_views"]
    assert sample.views_vchw.ndim == 4
    y_pred = predict_multiview_xyz(loaded, sample.views_vchw)
    assert len(y_pred) == 3
    result = run_packaged_m9_demo_inference(loaded)
    assert result.y_true is not None
    assert result.euclidean_error is not None
    assert result.euclidean_error < 50.0
