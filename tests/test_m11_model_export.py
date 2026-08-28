"""Tests for M11 single-view Fourier Hub model export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tomography_ml.localization.builders import materialize_lazy_modules
from tomography_ml.studies.single_view_m8 import make_m8_single_view_model
from tomography_ml_huggingface.model_export import (
    export_singleview_cnn_fourier,
    resolve_singleview_cnn_fourier_paths,
)


REPO = Path(__file__).resolve().parents[1]
STUDY_CKPT = REPO / "checkpoints" / "m8" / "m08_train_val_test_xyz.pt"


@pytest.mark.skipif(not STUDY_CKPT.is_file(), reason="M8 xyz study checkpoint absent")
def test_export_singleview_cnn_fourier_to_tmp(tmp_path: Path) -> None:
    local_toml = tmp_path / "local.toml"
    clone = tmp_path / "singleview_cnn_fourier"
    local_toml.write_text(
        "\n".join(
            [
                "[models.singleview_cnn_fourier]",
                'hub_id = "tbhugging/singleview_cnn_fourier"',
                'hub_url = "https://huggingface.co/tbhugging/singleview_cnn_fourier"',
                f'local_clone = "{clone}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = export_singleview_cnn_fourier(
        REPO,
        local_toml=local_toml,
        checkpoint_path=STUDY_CKPT,
    )
    assert result.weights_path.is_file()
    assert result.config_path.is_file()
    assert result.readme_path.is_file()
    cfg = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert cfg["architecture"] == "fourier"
    assert cfg["y_fields"] == ["particle_x", "particle_y", "particle_z"]
    assert cfg["n_params"] == result.n_params
    assert "validation_MSE" not in result.metrics
    assert "test_MSE" not in result.metrics
    assert "validation_RMSE_total" in result.metrics
    assert "test_RMSE_total" in result.metrics
    readme = result.readme_path.read_text(encoding="utf-8")
    assert "Euclidean" in readme
    assert "## What this model does" in readme
    assert "## Architecture" in readme
    assert "## Training configuration" in readme
    assert "## Evaluation Results" in readme
    assert "### Testing Data" in readme
    assert "### Metrics" in readme
    assert "### Results" in readme
    assert "## Held-out metrics" not in readme
    assert "## Contract" not in readme
    assert "datasets:" in readme
    assert "tbhugging/gummybear-tomography" in readme
    assert "https://github.com/tbgitoo/gummybear-tomography" in readme
    assert (
        "https://github.com/tbgitoo/gummybear-tomography/blob/master/"
        "GummyBearTomography_Final_Report.ipynb"
    ) in readme
    assert (
        "https://huggingface.co/datasets/tbhugging/gummybear-tomography/blob/main/"
        "checkpoints/m8/m08_train_val_test_xyz.pt"
    ) in readme
    assert "Fourier pooling in place of conventional" in readme
    assert "# libraries from https://github.com/tbgitoo/gummybear-tomography" in readme
    assert "## Inference" in readme
    assert "11_1_test_singleview_cnn_fourier.ipynb" in readme
    assert "validation_MSE" not in readme
    assert "pipeline_tag: image-feature-extraction" in readme
    assert "metrics:" in readme
    assert "- rmse" in readme
    assert "model-index:" in readme
    assert "RMSE_total (Euclidean xyz)" in readme
    assert "split: validation" in readme
    assert "split: test" in readme
    assert "| Split | Metric | Value |" in readme
    assert "model-cards#evaluation-results" in readme

    model = make_m8_single_view_model("fourier", n_outputs=3, device="cpu")
    materialize_lazy_modules(model, torch.zeros(1, 1, 128, 128))
    state = torch.load(result.weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    paths = resolve_singleview_cnn_fourier_paths(REPO, local_toml=local_toml)
    assert paths.local_clone == clone.resolve()
    assert paths.hub_id == "tbhugging/singleview_cnn_fourier"
