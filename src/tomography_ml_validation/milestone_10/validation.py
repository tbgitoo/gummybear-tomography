"""Installable contract tests for Milestone 10 hierarchical fusion."""

from __future__ import annotations

from pathlib import Path

import pytest

from tomography_ml.localization.localize_multiview import (
    FUSION_PATTERN_10_2,
    FUSION_PATTERN_10_2_POOLED,
)
from tomography_ml_validation.milestone_10.notebook_helpers import m10_corpus_paths


@pytest.mark.milestone("M10.3")
@pytest.mark.proves("10_2 hierarchical fusion keeps distinct Fourier vs pooled pattern IDs.")
def test_m10_2_fusion_pattern_ids():
    assert FUSION_PATTERN_10_2 != FUSION_PATTERN_10_2_POOLED
    assert "pooled" not in FUSION_PATTERN_10_2
    assert "pooled" in FUSION_PATTERN_10_2_POOLED


@pytest.mark.milestone("M10.3")
@pytest.mark.proves("M10 notebooks resolve illumination corpus plus checkpoint-policy keys.")
def test_m10_corpus_paths_resolve():
    repo = Path(__file__).resolve().parents[3]
    if not (repo / "pyproject.toml").is_file():
        pytest.skip("repository root not found from installed package")
    for mode in ("demo", "full"):
        paths = m10_corpus_paths(repo, data_mode=mode)
        assert paths["workbook_path"].name.endswith(".xlsx")
        assert "m10" in paths["output_root"].as_posix()
        assert "results_dir" in paths
        assert "batch_size" in paths
        assert "run_lr_study" in paths
        assert "load_existing" in paths
        assert "retrain" in paths
