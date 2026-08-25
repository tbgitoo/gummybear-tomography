"""Installable contract tests for Milestone 9 fusion ladders."""

from __future__ import annotations

from pathlib import Path

import pytest

from tomography_ml.localization.localize_multiview import (
    FUSION_PATTERN_09_0,
    FUSION_PATTERN_09_1,
    FUSION_PATTERN_09_1_POOLED,
    FUSION_PATTERN_09_2,
    FUSION_PATTERN_09_3,
    M9_2_FUSION_DEPTH,
    M9_2_FUSION_HIDDEN,
    M9_3_FUSION_DEPTH,
    M9_3_FUSION_HIDDEN,
)
from tomography_ml.studies.m9_expert_xyz_mean import assert_affine_identity_shared_linear
from tomography_ml_validation.milestone_09.notebook_helpers import m9_corpus_paths


@pytest.mark.milestone("M9.0")
@pytest.mark.proves(
    "Mean of Linear(h) equals Linear(mean(h)) for a shared WIN 3J head."
)
def test_m9_0_affine_identity_shared_linear():
    """Affine identity is a shared-trunk sanity check, not the 09_0 expert bank."""
    delta = assert_affine_identity_shared_linear()
    assert delta < 1e-5


@pytest.mark.milestone("M9.0")
@pytest.mark.proves("09_0 expert-mean fusion pattern is evaluation-time xyz averaging.")
def test_m9_0_fusion_pattern_is_expert_mean():
    assert FUSION_PATTERN_09_0 == "expert_xyz_mean"


@pytest.mark.milestone("M9.1")
@pytest.mark.proves("09_1 frozen fusion keeps distinct Fourier vs GAP pattern IDs.")
def test_m9_1_frozen_fusion_pattern_ids():
    assert FUSION_PATTERN_09_1 != FUSION_PATTERN_09_1_POOLED
    assert "pooled" not in FUSION_PATTERN_09_1


@pytest.mark.milestone("M9.2")
@pytest.mark.proves("09_3 is a larger fusion MLP than 09_2 under the same e2e protocol.")
def test_m9_2_compact_vs_09_3_capacity():
    assert M9_2_FUSION_HIDDEN == 128
    assert M9_2_FUSION_DEPTH == 1
    assert M9_3_FUSION_HIDDEN == 512
    assert M9_3_FUSION_DEPTH == 2
    assert FUSION_PATTERN_09_2 != FUSION_PATTERN_09_3


@pytest.mark.milestone("M9.0")
@pytest.mark.proves("M9 notebooks resolve the M8 corpus plus checkpoint-policy keys.")
def test_m9_corpus_paths_resolve():
    repo = Path(__file__).resolve().parents[3]
    if not (repo / "pyproject.toml").is_file():
        pytest.skip("repository root not found from installed package")
    for mode in ("demo", "full"):
        paths = m9_corpus_paths(repo, data_mode=mode)
        assert paths["workbook_path"].name.endswith(".xlsx")
        assert "m8" in paths["output_root"].as_posix()
        assert "results_dir" in paths
        assert "angle_stride_deg_09_0" in paths
        assert "load_existing" in paths
        assert "retrain" in paths
