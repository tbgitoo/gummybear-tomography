"""Installable contract tests for Milestone 8 freeze / grid records."""

from __future__ import annotations

from pathlib import Path

import pytest

import pandas as pd

from tomography_ml.localization import (
    win3b_receptive_field_grid,
    win3c_channel_capacity_grid,
    win3d_head_expressiveness_grid,
    win3e_architecture_freeze,
    win3f_selected_representation,
    win3g_selected_normalisation,
    win3h_optical_regime_grid,
    win3j_single_view_freeze,
)
from tomography_ml_validation.milestone_08.notebook_helpers import (
    assert_win3j_freeze_contract,
    m8_corpus_paths,
)


@pytest.mark.milestone("M8.3E")
@pytest.mark.proves(
    "Milestone 8 Step 3E architecture freeze records Fourier-base + MLP as primary."
)
def test_m8_3e_architecture_freeze_records_fourier_base_mlp():
    """Architecture freeze matches the inscribed post–3C retention."""
    freeze = win3e_architecture_freeze()
    assert freeze.selected_variant == "fourier_base_mlp"
    assert freeze.spatial_readout_type == "fourier_coded_pool"
    assert freeze.widths == (16, 32, 64)
    assert freeze.downsampling == "base"
    assert freeze.head == "mlp"
    assert freeze.positive_baseline == "flatten_base_mlp"
    assert freeze.negative_control == "pooled_base_base"
    assert freeze.library_class == "LocalizerSingleViewFourier"


@pytest.mark.milestone("M8.3J")
@pytest.mark.proves(
    "Milestone 8 Step 3J single-view freeze binds architecture, delta, and per-view z-score."
)
def test_m8_3j_single_view_freeze_binds_delta_and_per_view_zscore():
    """Single-view block freeze matches Final Report / plan Conclusion."""
    summary = assert_win3j_freeze_contract()
    assert summary["x_field"] == "anomaly_ref"
    assert summary["image_normalize"] == "per_image_zscore"
    assert summary["keep_angles_deg"] == 180.0


@pytest.mark.milestone("M8.3B")
@pytest.mark.proves("Milestone 8 Step 3B receptive-field grid is a small predefined set.")
def test_m8_3b_receptive_field_grid_is_bounded():
    """RF / downsampling grid stays small (no unbounded search)."""
    grid = win3b_receptive_field_grid()
    assert 3 <= len(grid) <= 12
    assert all(cfg.head_type in {"fourier", "flatten", "pooled"} for cfg in grid)


@pytest.mark.milestone("M8.3B")
@pytest.mark.proves("Milestone 8 Step 3B multi-seed aggregation produces mean/std columns.")
def test_m8_3b_aggregate_win3b_runs():
    from tomography_ml_validation.milestone_08.win3b_receptive_field import (
        aggregate_win3b_runs,
    )

    runs = pd.DataFrame(
        [
            {"variant": "fourier_base_base", "validation_RMSE_total": 1.0, "seed": 0},
            {"variant": "fourier_base_base", "validation_RMSE_total": 1.2, "seed": 1},
            {"variant": "flatten_base_base", "validation_RMSE_total": 0.8, "seed": 0},
        ]
    )
    summary = aggregate_win3b_runs(runs)
    assert len(summary) == 2
    row = summary.set_index("variant").loc["fourier_base_base"]
    assert row["validation_RMSE_total_mean"] == pytest.approx(1.1)
    assert row["validation_RMSE_total_std"] == pytest.approx(0.141421356, rel=1e-3)
    assert int(row["n_repeat"]) == 2


@pytest.mark.milestone("M8.3B")
@pytest.mark.proves("Milestone 8 Step 3B downsampling labels map to explicit MaxPool schedules.")
def test_m8_3b_describe_downsample_schedule():
    from tomography_ml.localization.encoder import describe_downsample_schedule

    base = describe_downsample_schedule(3, "base", height=128, width=128)
    assert base["maxpool_after_blocks"] == "none"
    assert base["feature_map_hw"] == (128, 128)

    medium = describe_downsample_schedule(3, "medium", height=128, width=128)
    assert medium["maxpool_after_blocks"] == "0,1"
    assert medium["feature_map_hw"] == (32, 32)

    high = describe_downsample_schedule(3, "high", height=128, width=128)
    assert high["maxpool_after_blocks"] == "0,1,2"
    assert high["feature_map_hw"] == (16, 16)


@pytest.mark.milestone("M8.3C")
@pytest.mark.proves("Milestone 8 Step 3C channel-capacity grid includes Fourier-base.")
def test_m8_3c_channel_capacity_includes_fourier_base():
    """Channel ladder retains Fourier-base among named variants."""
    names = {cfg.arch_name for cfg in win3c_channel_capacity_grid()}
    assert any("fourier" in name and "base" in name for name in names)


@pytest.mark.milestone("M8.3D")
@pytest.mark.proves("Milestone 8 Step 3D head grid compares linear vs MLP on the triad.")
def test_m8_3d_head_expressiveness_grid_nonempty():
    """Head expressiveness grid is non-empty and uses frozen geometry labels."""
    grid = win3d_head_expressiveness_grid()
    assert len(grid) >= 3
    assert any(cfg.flatten_head == "mlp" for cfg in grid)


@pytest.mark.milestone("M8.3F")
@pytest.mark.proves("Milestone 8 Step 3F selected representation is delta / anomaly_ref.")
def test_m8_3f_selected_representation_is_delta():
    """Capability path representation is the stored anomaly role."""
    rep = win3f_selected_representation()
    assert rep.name == "delta"
    assert rep.x_field == "anomaly_ref"


@pytest.mark.milestone("M8.3G")
@pytest.mark.proves("Milestone 8 Step 3G selected normalisation is per-view z-score.")
def test_m8_3g_selected_normalisation_is_per_image_zscore():
    """Downstream intensity normalisation matches the elevated standard."""
    norm = win3g_selected_normalisation()
    assert norm.image_normalize == "per_image_zscore"


@pytest.mark.milestone("M8.3H")
@pytest.mark.proves("Milestone 8 Step 3H optical-regime grid spans low / medium / high.")
def test_m8_3h_optical_regime_grid_spans_three_setups():
    """Optical regimes remain nuisance factors with three Excel setup ids."""
    regimes = win3h_optical_regime_grid()
    ids = {r.optical_setup_id for r in regimes}
    assert ids == {
        "opt_m8_low_001",
        "opt_m8_med_001",
        "opt_m8_high_001",
    }


@pytest.mark.milestone("M8.3I")
@pytest.mark.proves(
    "Milestone 8 Step 3I aggregates uppercase per-axis RMSE and plots non-empty bars."
)
def test_m8_3i_per_axis_plot_uses_uppercase_rmse_columns():
    """Training writes validation_RMSE_X/Y/Z; aggregator and plot must match."""
    from tomography_ml_validation.milestone_08.win3_study_common import aggregate_runs
    from tomography_ml_validation.milestone_08.win3i_observability import plot_win3i_per_axis

    runs = pd.DataFrame(
        [
            {
                "variant": "fourier_base_mlp",
                "validation_RMSE_total": 1.0,
                "validation_RMSE_X": 1.2,
                "validation_RMSE_Y": 0.9,
                "validation_RMSE_Z": 0.8,
                "seed": 0,
            },
            {
                "variant": "fourier_base_mlp",
                "validation_RMSE_total": 1.2,
                "validation_RMSE_X": 1.4,
                "validation_RMSE_Y": 1.1,
                "validation_RMSE_Z": 1.0,
                "seed": 1,
            },
        ]
    )
    summary = aggregate_runs(runs)
    row = summary.iloc[0]
    assert row["validation_RMSE_X_mean"] == pytest.approx(1.3)
    assert row["validation_RMSE_Y_mean"] == pytest.approx(1.0)
    assert row["validation_RMSE_Z_mean"] == pytest.approx(0.9)
    fig = plot_win3i_per_axis(summary)
    heights = [p.get_height() for p in fig.axes[0].patches]
    assert heights == pytest.approx([1.3, 1.0, 0.9])


@pytest.mark.milestone("M8.0")
@pytest.mark.proves("M8 workbook / output paths resolve for demo and full modes.")
def test_m8_0_corpus_paths_resolve(tmp_path: Path):
    """Path helper returns workbook and output roots under the repo layout."""
    # Use real repo root (package is installed from this tree).
    repo = Path(__file__).resolve().parents[3]
    if not (repo / "pyproject.toml").is_file():
        pytest.skip("repository root not found from installed package")
    for mode in ("demo", "full"):
        paths = m8_corpus_paths(repo, data_mode=mode)
        assert paths["workbook_path"].name.endswith(".xlsx")
        assert "m8" in paths["output_root"].as_posix()
