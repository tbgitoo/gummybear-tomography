"""Unit tests for the M9 09_0 expert-mean study helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from tomography_ml.localization.localize_multiview import (
    ExpertXyzMeanLocalizer,
    new_frozen_single_view_expert,
)
from tomography_ml.studies.m9_expert_xyz_mean import assert_affine_identity_shared_linear
from tomography_ml.studies.study_checkpoints import M09_EXPERT_XYZ_MEAN
from tomography_ml_validation.milestone_09.notebook_helpers import m9_corpus_paths
from tomography_ml_validation.plotting.m9_expert_xyz_mean import (
    collect_m9_0_bias_vs_std,
    plot_m9_0_bias_vs_expert_std,
    plot_m9_0_per_angle_experts,
)


class _OneSample:
    def __init__(self) -> None:
        self.rows = [type("R", (), {"sequence_id": "seq0"})()]
        self._x = torch.randn(2, 1, 8, 8)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx):
        return (
            {"anomaly_ref": self._x},
            {"particle_x": 0.1, "particle_y": 0.2, "particle_z": 0.3},
        )


def test_assert_affine_identity_shared_linear() -> None:
    delta = assert_affine_identity_shared_linear()
    assert delta < 1e-5


def test_m09_expert_checkpoint_filename() -> None:
    assert M09_EXPERT_XYZ_MEAN == "m09_expert_xyz_mean.pt"


def test_m9_0_plot_helpers() -> None:
    angles = (0.0, 90.0)
    bank = ExpertXyzMeanLocalizer(
        {theta: new_frozen_single_view_expert() for theta in angles}
    )
    ds = _OneSample()
    figs = plot_m9_0_per_angle_experts(
        bank=bank,
        dataset=ds,
        x_field="anomaly_ref",
        angles=angles,
        device="cpu",
        n_samples=1,
    )
    assert len(figs) == 1
    bias_df = collect_m9_0_bias_vs_std(
        bank=bank,
        dataset=ds,
        x_field="anomaly_ref",
        angles=angles,
        device="cpu",
    )
    assert set(bias_df["axis"]) == {"X", "Y", "Z"}
    fig_axes, fig_all = plot_m9_0_bias_vs_expert_std(bias_df)
    assert fig_axes is not None
    assert fig_all is not None


def test_m9_corpus_paths_keys() -> None:
    repo = Path(__file__).resolve().parents[1]
    paths = m9_corpus_paths(repo, data_mode="demo")
    assert paths["workbook_path"].name.endswith(".xlsx")
    assert paths["num_epochs"] == 40
    assert paths["angle_stride_deg_09_0"] == 90.0
    assert paths["load_existing"] is False
