"""Tests for centralized study checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

from tomography_ml.studies.study_checkpoints import (
    M08_TRAIN_VAL_TEST_Z,
    study_checkpoint_path,
    study_checkpoint_policy,
    study_results_dir,
)


def test_study_checkpoint_policy_full_vs_demo(tmp_path: Path) -> None:
    full = study_checkpoint_policy(
        repo_root=tmp_path,
        milestone="m8",
        data_mode="full",
        read_checkpoints=True,
    )
    assert full.enabled
    assert full.load_existing
    assert not full.retrain
    assert full.directory == tmp_path / "checkpoints" / "m8"

    demo = study_checkpoint_policy(
        repo_root=tmp_path,
        milestone="m8",
        data_mode="demo",
        read_checkpoints=True,
    )
    assert not demo.enabled
    assert demo.directory is None
    assert study_checkpoint_path(demo, M08_TRAIN_VAL_TEST_Z) is None

    overwrite = study_checkpoint_policy(
        repo_root=tmp_path,
        milestone="m9",
        data_mode="full",
        read_checkpoints=False,
    )
    assert overwrite.retrain
    assert not overwrite.load_existing
    assert study_checkpoint_path(overwrite, "m09_frozen_fourier_fusion.pt") == (
        tmp_path / "checkpoints" / "m9" / "m09_frozen_fourier_fusion.pt"
    )


def test_study_results_dir_fallback(tmp_path: Path) -> None:
    demo = study_checkpoint_policy(
        repo_root=tmp_path,
        milestone="m8",
        data_mode="demo",
        read_checkpoints=True,
    )
    fallback = tmp_path / "data" / "generated" / "m8_demo" / "_study"
    out = study_results_dir(demo, fallback=fallback)
    assert out == fallback
    assert out.is_dir()
