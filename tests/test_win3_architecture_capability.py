"""Smoke tests for parametrised Encode / builders / WIN 3 capability helpers."""

from __future__ import annotations

import torch

from tomography_ml.localization import (
    Encode,
    LocalizeSingleView,
    LocalizeSingleViewFlatten,
    LocalizerSingleViewFourier,
    SingleViewArchConfig,
    build_from_config,
    build_shared_subsets,
    count_parameters,
    default_mechanism_grid,
    describe_feature_geometry,
    make_flatten,
    make_pooled,
    materialize_lazy_modules,
    win3e_architecture_freeze,
    win3e_control_configs,
)
from tomography_ml.localization.architecture_capability import (
    collapse_to_mean_target,
    constant_mean_target_baseline_rmse,
    overfit_success,
    per_axis_rmse,
)
from tomography_ml.localization.encoder import CHANNEL_PRESETS


def test_encode_default_forward_features_shape() -> None:
    enc = Encode()
    feats = enc.forward_features(torch.randn(2, 1, 20, 24))
    assert feats.shape == (2, 64, 20, 24)


def test_encode_downsample_and_channel_presets() -> None:
    # ``medium`` = two 2× pools (legacy name was confusingly ``base``).
    enc = Encode(channels="narrow", downsample="medium")
    feats = enc.forward_features(torch.randn(1, 1, 32, 32))
    assert feats.shape == (1, 32, 8, 8)
    geom = describe_feature_geometry(enc, height=32, width=32)
    assert geom["flatten_length"] == 32 * 8 * 8
    # ``base`` downsample = WIN 3A (no MaxPool).
    enc_base = Encode(channels="base", downsample="base")
    feats_base = enc_base.forward_features(torch.randn(1, 1, 32, 32))
    assert feats_base.shape == (1, 64, 32, 32)


def test_encode_pre_flatten_channel_compression() -> None:
    enc = Encode(channels="base", downsample="medium", pre_flatten_channels=8)
    feats = enc.forward_features(torch.randn(2, 1, 16, 16))
    assert feats.shape[1] == 8
    assert enc.flatten_length(16, 16) == 8 * 4 * 4


def test_pooled_and_flatten_builders_forward_and_backward() -> None:
    x = torch.randn(3, 1, 16, 16, requires_grad=True)
    pooled = make_pooled(n_outputs=2, encoder_channels="base", downsample="base")
    flat = make_flatten(
        n_outputs=2,
        hidden=32,
        encoder_channels="base",
        downsample="medium",
        head_type="mlp",
    )
    materialize_lazy_modules(flat, torch.zeros(1, 1, 16, 16))
    yp = pooled(x)
    yf = flat(x)
    assert yp.shape == (3, 2)
    assert yf.shape == (3, 2)
    assert count_parameters(pooled) > 0
    assert count_parameters(flat) > 0
    (yp.sum() + yf.sum()).backward()
    assert x.grad is not None


def test_flatten_linear_head_and_geometry_log() -> None:
    model = make_flatten(
        n_outputs=1,
        encoder_channels=CHANNEL_PRESETS["base"],
        downsample="medium",
        head_type="linear",
        hidden=1,
    )
    dummy = torch.randn(1, 1, 24, 24)
    materialize_lazy_modules(model, dummy)
    geom = describe_feature_geometry(model.encoder, height=24, width=24)
    assert geom["flatten_length"] == model.encoder.out_channels * 6 * 6
    out = model(dummy)
    assert out.shape == (1, 1)
    assert count_parameters(model) < 5_000_000


def test_build_from_config_and_mechanism_grid_smoke() -> None:
    grid = default_mechanism_grid()
    assert len(grid) >= 5
    from tomography_ml.localization import (
        win3b_receptive_field_grid,
        win3c_channel_capacity_grid,
        win3d_head_expressiveness_grid,
    )

    b = win3b_receptive_field_grid()
    c = win3c_channel_capacity_grid()
    d = win3d_head_expressiveness_grid()
    assert all(cfg.head_type in {"fourier", "flatten"} for cfg in b)
    assert all(cfg.head_type in {"pooled", "fourier", "flatten"} for cfg in c)
    assert all(cfg.head_type in {"pooled", "fourier", "flatten"} for cfg in d)
    assert sum(1 for cfg in b if cfg.head_type == "fourier") >= 3
    assert sum(1 for cfg in c if cfg.head_type == "fourier") == 3
    assert any(cfg.arch_name == "pooled_base_base" for cfg in c)
    assert any(cfg.arch_name == "fourier_base_linear" for cfg in d)
    assert any(cfg.arch_name == "fourier_base_mlp" for cfg in d)
    assert any(cfg.arch_name == "flatten_base_linear" for cfg in d)
    assert any(cfg.arch_name == "flatten_base_mlp" for cfg in d)
    assert any(cfg.arch_name.startswith("flatten_") for cfg in b)
    assert any(cfg.arch_name.startswith("flatten_") for cfg in c)
    # ``base`` is WIN 3A for both axes.
    assert any(
        cfg.arch_name == "fourier_base_base" and cfg.downsample == "base" for cfg in b
    )
    cfg = SingleViewArchConfig(
        arch_name="unit_flatten",
        head_type="flatten",
        encoder_channels=(8, 16),
        downsample="low",
        flatten_hidden=16,
        flatten_head="mlp",
    )
    model = build_from_config(cfg, n_outputs=3)
    materialize_lazy_modules(model, torch.zeros(1, 1, 16, 16))
    assert model(torch.randn(2, 1, 16, 16)).shape == (2, 3)


def test_win3e_architecture_freeze_matches_default_localizer() -> None:
    freeze = win3e_architecture_freeze()
    assert freeze.selected_variant == "fourier_base_mlp"
    assert freeze.spatial_readout_type == "fourier_coded_pool"
    assert freeze.widths == (16, 32, 64)
    assert freeze.downsampling == "base"
    assert freeze.head == "mlp"
    assert freeze.library_class == "LocalizerSingleViewFourier"

    primary, positive, negative = win3e_control_configs()
    assert primary.arch_name == freeze.selected_variant
    assert positive.arch_name == freeze.positive_baseline
    assert negative.arch_name == freeze.negative_control

    dummy = torch.zeros(1, 1, 16, 16)
    default = LocalizerSingleViewFourier(n_outputs=3)
    built = build_from_config(freeze.primary_config(), n_outputs=3)
    materialize_lazy_modules(built, dummy)
    assert count_parameters(default) == count_parameters(built)
    assert tuple(default.encoder.channels) == freeze.widths
    assert default.encoder.downsample == freeze.downsampling
    assert default.hidden == freeze.head_hidden
    assert default(torch.randn(2, 1, 16, 16)).shape == (2, 3)


def test_win3f_representation_grid() -> None:
    from tomography_ml.localization import (
        win3f_representation_grid,
        win3f_selected_representation,
        win3g_normalisation_grid,
        win3g_selected_normalisation,
        win3h_optical_regime_grid,
    )

    reps = win3f_representation_grid()
    assert [r.name for r in reps] == ["delta", "clean", "observed"]
    assert [r.x_field for r in reps] == [
        "anomaly_ref",
        "clean_ref",
        "observed_ref",
    ]
    selected = win3f_selected_representation()
    assert selected.name == "delta"
    assert selected.x_field == "anomaly_ref"

    norms = win3g_normalisation_grid()
    assert [n.name for n in norms] == [
        "raw",
        "train_split_zscore",
        "per_image_zscore",
        "per_image_minmax",
    ]
    assert [n.image_normalize for n in norms] == [
        "none",
        "train_split_zscore",
        "per_image_zscore",
        "per_image_minmax",
    ]
    assert [n.diagnostic for n in norms] == [False, False, True, True]
    selected_norm = win3g_selected_normalisation()
    assert selected_norm.name == "per_image_zscore"
    assert selected_norm.image_normalize == "per_image_zscore"

    regimes = win3h_optical_regime_grid()
    assert [r.name for r in regimes] == ["low", "medium", "high"]
    assert [r.optical_setup_id for r in regimes] == [
        "opt_m8_low_001",
        "opt_m8_med_001",
        "opt_m8_high_001",
    ]
    assert regimes[0].mu_a < regimes[1].mu_a < regimes[2].mu_a
    assert regimes[0].mu_s < regimes[1].mu_s < regimes[2].mu_s

    from tomography_ml.localization import win3i_key_result_sources

    sources = win3i_key_result_sources()
    assert [s["win"] for s in sources] == ["3F", "3G", "3H"]
    assert all("relative_csv" in s for s in sources)

    from tomography_ml.localization import (
        SingleViewBlockFreezeRecord,
        win3j_single_view_freeze,
    )

    block = win3j_single_view_freeze()
    assert isinstance(block, SingleViewBlockFreezeRecord)
    assert block.architecture.selected_variant == "fourier_base_mlp"
    assert block.representation_name == "delta"
    assert block.x_field == "anomaly_ref"
    assert block.image_normalize == "per_image_zscore"
    assert block.normalisation().image_normalize == "per_image_zscore"
    assert block.lr_by_role()["primary"] == 0.03
    assert "architecture" in block.freeze_fields
    assert "normalisation" in block.freeze_fields


def test_shared_subsets_are_reproducible_and_fixed() -> None:
    seqs = [f"s{i}" for i in range(12)]
    a = build_shared_subsets(
        n_pool=12,
        sequence_ids=seqs,
        ns=(1, 2, 5),
        n_reps=2,
        base_seed=7,
    )
    b = build_shared_subsets(
        n_pool=12,
        sequence_ids=seqs,
        ns=(1, 2, 5),
        n_reps=2,
        base_seed=7,
    )
    assert a == b
    assert {s.n for s in a} == {1, 2, 5}
    assert all(len(s.indices) == s.n for s in a)


def test_rmse_baseline_collapse_and_overfit_helpers() -> None:
    target = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
    pred_good = target.clone()
    pred_mid = torch.ones_like(target)
    rmse = per_axis_rmse(pred_good, target, ("particle_x", "particle_z"))
    assert rmse["train_RMSE_total"] == 0.0
    assert rmse["train_RMSE_X"] == 0.0
    assert rmse["train_RMSE_Z"] == 0.0
    baseline = constant_mean_target_baseline_rmse(target)
    assert baseline > 0
    assert collapse_to_mean_target(pred_mid, target) is True
    assert collapse_to_mean_target(pred_good, target) is False
    assert overfit_success(train_rmse_total=0.1, baseline_rmse=1.0) is True
    assert overfit_success(train_rmse_total=0.9, baseline_rmse=1.0) is False


def test_legacy_localizer_classes_still_construct() -> None:
    assert LocalizeSingleView(Encode(), n_outputs=1)(
        torch.randn(2, 1, 8, 8)
    ).shape == (2, 1)
    model = LocalizeSingleViewFlatten(Encode(), n_outputs=1, hidden=8)
    assert model(torch.randn(2, 1, 8, 8)).shape == (2, 1)
