"""M9/M10 09_0 / 09_1 / 09_2 multi-view smoke tests."""

from __future__ import annotations

import math

import torch

from tomography_ml.localization.localize_multiview import (
    FUSION_PATTERN_09_0,
    FUSION_PATTERN_09_1,
    FUSION_PATTERN_09_1_DEEPSETS_FOURIER,
    FUSION_PATTERN_09_1_DEEPSETS_NO_FOURIER,
    FUSION_PATTERN_09_1_MEAN_POOL,
    FUSION_PATTERN_09_1_MEAN_POOL_POOLED,
    FUSION_PATTERN_09_1_POOLED,
    FUSION_PATTERN_09_2,
    FUSION_PATTERN_09_2_POOLED,
    FUSION_PATTERN_09_3,
    FUSION_PATTERN_09_3_POOLED,
    FUSION_PATTERN_10_BASELINE,
    FUSION_PATTERN_10_BASELINE_POOLED,
    FUSION_PATTERN_10_1_C,
    FUSION_PATTERN_10_1_C_POOLED,
    FUSION_PATTERN_10_1_C_FROZEN,
    FUSION_PATTERN_10_1_C_FROZEN_POOLED,
    FUSION_PATTERN_10_1_D,
    FUSION_PATTERN_10_1_D_POOLED,
    FUSION_PATTERN_10_1_D_FROZEN,
    FUSION_PATTERN_10_1_D_FROZEN_POOLED,
    FUSION_PATTERN_MEAN_LATENT_SANITY,
    M9_2_FUSION_DEPTH,
    M9_2_FUSION_HIDDEN,
    M9_3_FUSION_DEPTH,
    M9_3_FUSION_HIDDEN,
    M10_LIGHT_ANGLES_DEG,
    PACKING_MEAN_POOL,
    PACKING_ORDERED_CONCAT,
    CompactLatentFusionLocalizer,
    DeepSetsFusionHead,
    ExpertXyzMeanLocalizer,
    FrozenEncoderDeepSetsLocalizer,
    GeometryAwareFourierFusionLocalizer,
    MeanLatentFusionLocalizer,
    make_angle_features,
    match_expert_index,
    mean_coordinates,
    new_frozen_pooled_single_view_expert,
    new_frozen_single_view_expert,
    pack_geometry_tokens,
    shared_xyz_mean,
)
from tomography_ml.localization.localizer import LocalizerSingleViewFourier


def test_make_angle_features_sin_cos() -> None:
    feats = make_angle_features(torch.tensor([0.0, 90.0, 180.0]))
    assert feats.shape == (3, 2)
    assert torch.allclose(feats[0], torch.tensor([0.0, 1.0]), atol=1e-5)
    assert torch.allclose(feats[1], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(feats[2], torch.tensor([0.0, -1.0]), atol=1e-5)


def test_mean_coordinates() -> None:
    xyz = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]],
        ]
    )
    mean = mean_coordinates(xyz)
    assert torch.allclose(mean, torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]))


def test_match_expert_index() -> None:
    assert match_expert_index(180.0, (0.0, 90.0, 180.0, 270.0)) == 2


def test_expert_xyz_mean_forward() -> None:
    experts = {
        0.0: new_frozen_single_view_expert(),
        90.0: new_frozen_single_view_expert(),
    }
    bank = ExpertXyzMeanLocalizer(experts)
    assert bank.fusion_pattern == FUSION_PATTERN_09_0
    assert isinstance(bank.expert_for_angle(0.0), LocalizerSingleViewFourier)

    views = torch.randn(2, 2, 1, 16, 16)
    angles = torch.tensor([0.0, 90.0])
    per_view = bank.predict_per_view(views, angles)
    assert per_view.shape == (2, 2, 3)

    fused = bank(views, angles)
    assert fused.shape == (2, 3)
    assert torch.allclose(fused, per_view.mean(dim=1))

    meta = bank.describe()
    assert meta["n_experts"] == 2
    assert meta["fusion_pattern"] == FUSION_PATTERN_09_0
    assert meta["learned_fusion_module"] is False
    assert meta["learned_parameter_count"] == bank.learned_parameter_count()
    assert bank.learned_parameter_count() > 0


def test_expert_xyz_mean_defaults_angles_when_v_matches() -> None:
    bank = ExpertXyzMeanLocalizer(
        {
            0.0: new_frozen_single_view_expert(),
            180.0: new_frozen_single_view_expert(),
        }
    )
    views = torch.randn(1, 2, 1, 12, 12)
    out = bank(views)  # angles inferred from expert_angles_deg
    assert out.shape == (1, 3)


def test_new_frozen_expert_matches_primary_class() -> None:
    model = new_frozen_single_view_expert()
    assert isinstance(model, LocalizerSingleViewFourier)
    y = model(torch.randn(3, 1, 8, 8))
    assert y.shape == (3, 3)
    assert math.isfinite(float(y.sum()))


def test_encode_latent_then_predict_matches_forward() -> None:
    model = new_frozen_single_view_expert()
    x = torch.randn(4, 1, 16, 16)
    h = model.encode_latent(x)
    assert h.shape == (4, model.hidden)
    y = model.predict_from_latent(h)
    assert torch.allclose(y, model(x), atol=1e-5)


def test_mean_latent_sanity_affine_identity() -> None:
    """Demoted path: mean(Linear(h_j)) == Linear(mean(h_j))."""
    backbone = new_frozen_single_view_expert()
    sanity = MeanLatentFusionLocalizer(backbone, freeze_encoder=False)
    assert sanity.fusion_pattern == FUSION_PATTERN_MEAN_LATENT_SANITY
    views = torch.randn(3, 6, 1, 12, 12)
    xyz_mean = shared_xyz_mean(backbone, views)
    out = sanity(views)
    assert torch.allclose(xyz_mean, out, atol=1e-5, rtol=1e-5)


def test_compact_09_1_forward_and_frozen_encoder() -> None:
    backbone = new_frozen_single_view_expert()
    # Mark a backbone weight so we can check it stays frozen / unchanged.
    enc_param = next(backbone.encoder.parameters())
    before = enc_param.detach().clone()

    f1 = CompactLatentFusionLocalizer(
        backbone, n_views=6, fusion_hidden=64, freeze_encoder=True
    )
    assert f1.fusion_pattern == FUSION_PATTERN_09_1
    assert f1.packing == PACKING_ORDERED_CONCAT
    assert f1.describe()["learned_fusion_module"] is True
    assert f1.describe()["packing"] == PACKING_ORDERED_CONCAT
    assert f1.learned_parameter_count() > 0
    # Encoder frozen: only fusion params trainable.
    assert all(not p.requires_grad for p in backbone.parameters())
    assert all(p.requires_grad for p in f1.fusion.parameters())

    views = torch.randn(2, 6, 1, 16, 16)
    out = f1(views)
    assert out.shape == (2, 3)
    h = f1.encode_view_latents(views)
    assert h.shape == (2, 6, backbone.hidden)
    packed = f1.pack_latents(h)
    assert packed.shape == (2, 6 * backbone.hidden)

    # One training step should not attach grads to frozen encoder weights.
    loss = out.sum()
    loss.backward()
    for p in f1.fusion.parameters():
        assert p.grad is not None
    assert enc_param.grad is None
    assert torch.equal(enc_param.detach(), before)


def test_compact_09_1_mean_pool_packing() -> None:
    """mean_pool uses the same MLP recipe; only latent aggregation differs."""
    backbone = new_frozen_single_view_expert()
    n_views = 6
    concat = CompactLatentFusionLocalizer(
        backbone,
        n_views=n_views,
        fusion_hidden=64,
        freeze_encoder=True,
        packing=PACKING_ORDERED_CONCAT,
    )
    # Fresh trunk so freeze on concat does not affect mean_pool construction.
    trunk = new_frozen_single_view_expert()
    mean_pool = CompactLatentFusionLocalizer(
        trunk,
        n_views=n_views,
        fusion_hidden=64,
        freeze_encoder=True,
        packing=PACKING_MEAN_POOL,
    )
    assert mean_pool.fusion_pattern == FUSION_PATTERN_09_1_MEAN_POOL
    assert mean_pool.packing == PACKING_MEAN_POOL
    assert mean_pool.describe()["packing"] == PACKING_MEAN_POOL
    assert mean_pool.fusion[0].in_features == trunk.hidden
    assert concat.fusion[0].in_features == n_views * backbone.hidden
    assert mean_pool.learned_parameter_count() < concat.learned_parameter_count()

    views = torch.randn(2, n_views, 1, 12, 12)
    out = mean_pool(views)
    assert out.shape == (2, 3)
    h = mean_pool.encode_view_latents(views)
    packed = mean_pool.pack_latents(h)
    assert packed.shape == (2, trunk.hidden)
    assert torch.allclose(packed, h.mean(dim=1), atol=1e-6)

    loss = out.sum()
    loss.backward()
    for p in mean_pool.fusion.parameters():
        assert p.grad is not None
    assert next(trunk.encoder.parameters()).grad is None


def test_compact_09_1_pooled_backbone_patterns() -> None:
    """Pooled GAP trunk uses *_frozen_pooled fusion patterns."""
    trunk = new_frozen_pooled_single_view_expert()
    concat = CompactLatentFusionLocalizer(
        trunk, n_views=4, fusion_hidden=32, freeze_encoder=True
    )
    assert concat.fusion_pattern == FUSION_PATTERN_09_1_POOLED
    assert concat.backbone_kind == "pooled"
    assert concat.describe()["variant_id"] == (
        "m09_1_compact_fusion_mlp_frozen_pooled"
    )
    assert concat.describe()["latent_cut"] == "gap_embed_relu"

    trunk2 = new_frozen_pooled_single_view_expert()
    mean_pool = CompactLatentFusionLocalizer(
        trunk2,
        n_views=4,
        fusion_hidden=32,
        freeze_encoder=True,
        packing=PACKING_MEAN_POOL,
    )
    assert mean_pool.fusion_pattern == FUSION_PATTERN_09_1_MEAN_POOL_POOLED
    assert mean_pool.describe()["variant_id"] == (
        "m09_1_compact_fusion_mlp_mean_pool_frozen_pooled"
    )
    views = torch.randn(2, 4, 1, 12, 12)
    assert concat(views).shape == (2, 3)
    assert mean_pool(views).shape == (2, 3)


def test_compact_09_1_rejects_bad_packing() -> None:
    try:
        CompactLatentFusionLocalizer(
            new_frozen_single_view_expert(),
            n_views=4,
            packing="attention",
        )
        raise AssertionError("expected ValueError for unknown packing")
    except ValueError as exc:
        assert "packing" in str(exc)


def test_compact_09_1_rejects_wrong_view_count() -> None:
    f1 = CompactLatentFusionLocalizer(
        new_frozen_single_view_expert(), n_views=4, freeze_encoder=True
    )
    try:
        f1(torch.randn(1, 3, 1, 8, 8))
        raise AssertionError("expected ValueError for wrong V")
    except ValueError as exc:
        assert "V=4" in str(exc) or "V=3" in str(exc)


def test_deepsets_fusion_head_shape_and_permutation_invariance() -> None:
    head = DeepSetsFusionHead(latent_dim=16, phi_hidden=8, rho_hidden=8)
    h = torch.randn(3, 5, 16)
    out = head(h)
    assert out.shape == (3, 3)
    perm = h[:, [4, 1, 0, 3, 2], :]
    assert torch.allclose(head(h), head(perm), atol=1e-5, rtol=1e-5)


def test_deepsets_fourier_frozen_encoder() -> None:
    model = FrozenEncoderDeepSetsLocalizer.for_09_1_fourier(
        n_views=4, phi_hidden=32, rho_hidden=32
    )
    assert model.fusion_pattern == FUSION_PATTERN_09_1_DEEPSETS_FOURIER
    assert model.describe()["display_label"] == "DeepSets Fourier"
    assert model.freeze_encoder is True
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    views = torch.randn(2, 4, 1, 12, 12)
    out = model(views)
    assert out.shape == (2, 3)
    out.sum().backward()
    assert next(model.backbone.encoder.parameters()).grad is None
    for p in model.head.parameters():
        assert p.grad is not None


def test_deepsets_no_fourier_frozen_encoder() -> None:
    model = FrozenEncoderDeepSetsLocalizer.for_09_1_no_fourier(
        n_views=4, phi_hidden=32, rho_hidden=32
    )
    assert model.fusion_pattern == FUSION_PATTERN_09_1_DEEPSETS_NO_FOURIER
    assert model.describe()["display_label"] == "DeepSets no-Fourier"
    assert model.backbone_kind == "pooled"
    assert all(not p.requires_grad for p in model.backbone.parameters())
    views = torch.randn(2, 4, 1, 12, 12)
    out = model(views)
    assert out.shape == (2, 3)
    h = model.encode_view_latents(views)
    assert h.shape == (2, 4, model.latent_dim)


def test_pack_geometry_tokens() -> None:
    h = torch.randn(2, 3, 8)
    angles = torch.tensor([[0.0, 90.0, 180.0], [0.0, 90.0, 180.0]])
    tokens = pack_geometry_tokens(h, angles)
    assert tokens.shape == (2, 3, 10)
    assert torch.allclose(tokens[..., :8], h)
    assert torch.allclose(
        tokens[0, :, 8:], make_angle_features(angles[0]), atol=1e-5
    )


def test_09_2_geometry_aware_e2e_forward_and_grads() -> None:
    angles = (0.0, 60.0, 120.0, 180.0)
    backbone = new_frozen_single_view_expert()
    enc_param = next(backbone.encoder.parameters())

    f2 = GeometryAwareFourierFusionLocalizer.for_09_2(
        backbone,
        n_views=len(angles),
        view_angles_deg=angles,
    )
    assert f2.fusion_pattern == FUSION_PATTERN_09_2
    assert f2.fusion_depth == M9_2_FUSION_DEPTH
    assert f2.fusion_hidden == M9_2_FUSION_HIDDEN
    assert f2.describe()["encoder_frozen"] is False
    assert f2.describe()["end_to_end"] is True
    assert f2.describe()["geometry_features"] == ("sin_theta", "cos_theta")
    assert all(p.requires_grad for p in backbone.parameters())
    assert all(p.requires_grad for p in f2.fusion.parameters())

    views = torch.randn(2, len(angles), 1, 16, 16)
    # Default angles path (train_full_split compatibility).
    out = f2(views)
    assert out.shape == (2, 3)
    # Explicit angles path.
    out2 = f2(views, torch.tensor(list(angles)))
    assert torch.allclose(out, out2, atol=1e-5)

    loss = out.sum()
    loss.backward()
    assert enc_param.grad is not None
    for p in f2.fusion.parameters():
        assert p.grad is not None


def test_09_2_09_3_pooled_geometry_variants() -> None:
    """Pooled GAP trunk e2e + geometry (09_2B / 09_3 capacity axis)."""
    angles = (0.0, 90.0, 180.0, 270.0)
    f2p = GeometryAwareFourierFusionLocalizer.for_09_2_pooled(
        n_views=len(angles),
        view_angles_deg=angles,
    )
    f3p = GeometryAwareFourierFusionLocalizer.for_09_3_pooled(
        n_views=len(angles),
        view_angles_deg=angles,
    )
    assert f2p.fusion_pattern == FUSION_PATTERN_09_2_POOLED
    assert f3p.fusion_pattern == FUSION_PATTERN_09_3_POOLED
    assert f2p.backbone_kind == "pooled"
    assert f3p.backbone_kind == "pooled"
    assert f2p.describe()["latent_cut"] == "gap_embed_relu"
    assert f3p.describe()["variant_id"] == (
        "m09_3_e2e_pooled_geometry_large_fusion"
    )
    assert f3p.fusion_parameter_count() > f2p.fusion_parameter_count()
    views = torch.randn(2, len(angles), 1, 12, 12)
    assert f2p(views).shape == (2, 3)
    assert f3p(views).shape == (2, 3)
    f2p(views).sum().backward()
    assert next(f2p.backbone.parameters()).grad is not None


def test_09_2_rejects_angle_count_mismatch() -> None:
    try:
        GeometryAwareFourierFusionLocalizer(
            n_views=4,
            view_angles_deg=(0.0, 90.0),
        )
        raise AssertionError("expected ValueError for angle count mismatch")
    except ValueError as exc:
        assert "n_views" in str(exc)


def test_09_3_large_fusion_capacity_axis() -> None:
    angles = (0.0, 90.0, 180.0, 270.0)
    f2 = GeometryAwareFourierFusionLocalizer.for_09_2(
        n_views=len(angles),
        view_angles_deg=angles,
    )
    f3 = GeometryAwareFourierFusionLocalizer.for_09_3(
        n_views=len(angles),
        view_angles_deg=angles,
    )
    assert f3.fusion_pattern == FUSION_PATTERN_09_3
    assert f3.fusion_hidden == M9_3_FUSION_HIDDEN
    assert f3.fusion_depth == M9_3_FUSION_DEPTH
    assert f3.fusion_hidden > f2.fusion_hidden
    assert f3.fusion_depth >= f2.fusion_depth
    assert f3.fusion_parameter_count() > f2.fusion_parameter_count()
    assert f3.describe()["variant_id"].startswith("m09_3_")

    views = torch.randn(2, len(angles), 1, 12, 12)
    out = f3(views)
    assert out.shape == (2, 3)
    enc_param = next(f3.backbone.encoder.parameters())
    out.sum().backward()
    assert enc_param.grad is not None


def test_10_baseline_illumination_geometry_tokens_and_forward() -> None:
    from tomography_ml.localization.localize_multiview import (
        M10_LIGHT_ANGLES_DEG,
        light_angle_deg_from_optical_setup_id,
        light_xy_from_angle_deg,
        pack_illumination_geometry_tokens,
    )

    assert light_angle_deg_from_optical_setup_id("opt_m10_illum_120") == 120.0
    x, y, z = light_xy_from_angle_deg(90.0)
    assert abs(x) < 1e-6 and abs(y - 20.0) < 1e-6 and z == 10.0
    assert M10_LIGHT_ANGLES_DEG[1] == 60.0

    angles = (0.0, 90.0, 180.0, 270.0)
    f4 = GeometryAwareFourierFusionLocalizer.for_10_baseline(
        n_views=len(angles),
        view_angles_deg=angles,
    )
    assert f4.fusion_pattern == FUSION_PATTERN_10_BASELINE
    assert f4.include_illumination is True
    assert f4.fusion_hidden == M9_2_FUSION_HIDDEN
    assert f4.describe()["geometry_features"] == (
        "sin_camera",
        "cos_camera",
        "sin_light",
        "cos_light",
    )

    views = torch.randn(2, len(angles), 1, 12, 12)
    light = torch.tensor([0.0, 180.0])
    out = f4(views, light_angles_deg=light)
    assert out.shape == (2, 3)

    h = f4.encode_view_latents(views)
    cam = torch.tensor([list(angles), list(angles)], dtype=torch.float32)
    tokens = pack_illumination_geometry_tokens(
        h, cam, light.unsqueeze(1).expand(2, len(angles))
    )
    assert tokens.shape == (2, len(angles), h.shape[-1] + 4)

    try:
        f4(views)  # missing light
        raise AssertionError("expected ValueError when light_angles_deg missing")
    except ValueError as exc:
        assert "light_angles_deg" in str(exc)

    f4p = GeometryAwareFourierFusionLocalizer.for_10_baseline_pooled(
        n_views=len(angles),
        view_angles_deg=angles,
    )
    assert f4p.fusion_pattern == FUSION_PATTERN_10_BASELINE_POOLED
    assert f4p.include_illumination is True
    assert f4p.backbone_kind == "pooled"
    assert f4p.describe()["latent_cut"] == "gap_embed_relu"
    assert f4p.describe()["geometry_features"] == (
        "sin_camera",
        "cos_camera",
        "sin_light",
        "cos_light",
    )
    out_p = f4p(views, light_angles_deg=light)
    assert out_p.shape == (2, 3)
    enc_p = next(f4p.backbone.encoder.parameters())
    out_p.sum().backward()
    assert enc_p.grad is not None


def test_10_1_illumination_only_factories() -> None:
    """10_1-C (no geom) and 10_1-D (light sin/cos) at V=lights."""
    lights = M10_LIGHT_ANGLES_DEG
    n = len(lights)
    c = CompactLatentFusionLocalizer.for_10_1_c(n_views=n)
    assert c.fusion_pattern == FUSION_PATTERN_10_1_C
    assert c.freeze_encoder is False
    assert c.describe()["end_to_end"] is True

    d = GeometryAwareFourierFusionLocalizer.for_10_1_d(
        n_views=n,
        light_angles_deg=lights,
    )
    assert d.fusion_pattern == FUSION_PATTERN_10_1_D
    assert d.include_illumination is False
    assert d.describe()["geometry_features"] == ("sin_light", "cos_light")
    assert list(d.view_angles_deg) == list(lights)

    views = torch.randn(2, n, 1, 12, 12)
    out_c = c(views)
    out_d = d(views)
    assert out_c.shape == (2, 3)
    assert out_d.shape == (2, 3)
    enc = next(c.backbone.encoder.parameters())
    out_c.sum().backward()
    assert enc.grad is not None

    c_p = CompactLatentFusionLocalizer.for_10_1_c_pooled(n_views=n)
    assert c_p.fusion_pattern == FUSION_PATTERN_10_1_C_POOLED
    assert c_p.backbone_kind == "pooled"
    assert c_p.describe()["latent_cut"] == "gap_embed_relu"
    d_p = GeometryAwareFourierFusionLocalizer.for_10_1_d_pooled(
        n_views=n,
        light_angles_deg=lights,
    )
    assert d_p.fusion_pattern == FUSION_PATTERN_10_1_D_POOLED
    assert d_p.backbone_kind == "pooled"
    assert d_p.describe()["geometry_features"] == ("sin_light", "cos_light")
    out_cp = c_p(views)
    out_dp = d_p(views)
    assert out_cp.shape == (2, 3)
    assert out_dp.shape == (2, 3)
    enc_p = next(c_p.backbone.encoder.parameters())
    out_cp.sum().backward()
    assert enc_p.grad is not None

    c_f = CompactLatentFusionLocalizer.for_10_1_c_frozen(n_views=n)
    assert c_f.fusion_pattern == FUSION_PATTERN_10_1_C_FROZEN
    assert c_f.freeze_encoder is True
    assert c_f.describe()["end_to_end"] is False
    d_f = GeometryAwareFourierFusionLocalizer.for_10_1_d_frozen(
        n_views=n,
        light_angles_deg=lights,
    )
    assert d_f.fusion_pattern == FUSION_PATTERN_10_1_D_FROZEN
    assert d_f.freeze_encoder is True
    assert not any(p.requires_grad for p in d_f.backbone.parameters())
    assert any(p.requires_grad for p in d_f.fusion.parameters())
    out_cf = c_f(views)
    out_df = d_f(views)
    assert out_cf.shape == (2, 3)
    assert out_df.shape == (2, 3)
    # Frozen trunk: no backbone grads from fusion-only loss
    enc_f = next(c_f.backbone.encoder.parameters())
    assert enc_f.requires_grad is False
    out_cf.sum().backward()
    assert enc_f.grad is None or float(enc_f.grad.abs().sum()) == 0.0

    c_fp = CompactLatentFusionLocalizer.for_10_1_c_frozen_pooled(n_views=n)
    d_fp = GeometryAwareFourierFusionLocalizer.for_10_1_d_frozen_pooled(
        n_views=n,
        light_angles_deg=lights,
    )
    assert c_fp.fusion_pattern == FUSION_PATTERN_10_1_C_FROZEN_POOLED
    assert d_fp.fusion_pattern == FUSION_PATTERN_10_1_D_FROZEN_POOLED
    assert c_fp(views).shape == (2, 3)
    assert d_fp(views).shape == (2, 3)

    # FiLM angle conditioning (preferred 10_1A-D default when enabled).
    from tomography_ml.localization.localize_multiview import (
        GEOMETRY_MODE_FILM,
        AngleConditionFiLM,
    )

    d_film = GeometryAwareFourierFusionLocalizer.for_10_1_d_frozen(
        n_views=n,
        light_angles_deg=lights,
        geometry_mode=GEOMETRY_MODE_FILM,
    )
    assert d_film.geometry_mode == GEOMETRY_MODE_FILM
    assert isinstance(d_film.angle_film, AngleConditionFiLM)
    assert d_film.describe()["geometry_mode"] == GEOMETRY_MODE_FILM
    assert d_film.describe()["token_dim"] == int(d_film.backbone.hidden)
    assert "FiLM" in d_film.describe()["note"]
    out_film = d_film(views)
    assert out_film.shape == (2, 3)
    # Identity FiLM init: film-modulated forward stays finite.
    assert torch.isfinite(out_film).all()
    assert any(p.requires_grad for p in d_film.angle_film.parameters())
    assert not any(p.requires_grad for p in d_film.backbone.parameters())

    d_film_p = GeometryAwareFourierFusionLocalizer.for_10_1_d_frozen_pooled(
        n_views=n,
        light_angles_deg=lights,
        geometry_mode=GEOMETRY_MODE_FILM,
    )
    assert d_film_p(views).shape == (2, 3)
    assert d_film_p.describe()["geometry_mode"] == GEOMETRY_MODE_FILM


def test_10_2_hierarchical_light_then_camera() -> None:
    """10_2: fuse lights within each camera, then fuse cameras."""
    from tomography_ml.localization.localize_multiview import (
        M10_2_CAMERA_LATENT_DIM,
        M10_2_FUSION_HIDDEN,
        FUSION_PATTERN_10_2,
        HierarchicalLightThenCameraFusionLocalizer,
        ensure_camera_light_views,
    )

    cams = (0.0, 90.0, 180.0, 270.0)
    lights = M10_LIGHT_ANGLES_DEG
    model = HierarchicalLightThenCameraFusionLocalizer.for_10_2(
        n_cameras=len(cams),
        n_lights=len(lights),
        camera_angles_deg=cams,
        light_angles_deg=lights,
        flat_layout="light_major",
    )
    assert model.fusion_pattern == FUSION_PATTERN_10_2
    assert model.fusion_hidden == M10_2_FUSION_HIDDEN
    assert model.camera_latent_dim == M10_2_CAMERA_LATENT_DIM
    assert model.describe()["packing"] == "hierarchical_light_then_camera"
    assert model.flat_layout == "light_major"

    # Canonical 6-D input [B, I, V, C, H, W]
    views6 = torch.randn(2, len(lights), len(cams), 1, 12, 12)
    out6 = model(views6)
    assert out6.shape == (2, 3)

    # Light-major flat (canonical flatten of [I, V, ...])
    views_lm = views6.reshape(2, len(lights) * len(cams), 1, 12, 12)
    out_lm = model(views_lm, layout="light_major")
    assert out_lm.shape == (2, 3)
    assert torch.allclose(out6, out_lm, atol=1e-5)

    # Camera-major flat (legacy) matches after reshape
    cam_major_6d = views6.permute(0, 2, 1, 3, 4, 5)
    views_cm = cam_major_6d.reshape(2, len(cams) * len(lights), 1, 12, 12)
    grid = ensure_camera_light_views(
        views_cm,
        n_cameras=len(cams),
        n_lights=len(lights),
        layout="camera_major",
    )
    assert grid.shape == (2, len(cams), len(lights), 1, 12, 12)
    out_cm = model(views_cm, layout="camera_major")
    assert out_cm.shape == (2, 3)
    assert torch.allclose(out6, out_cm, atol=1e-5)

    enc = next(model.backbone.encoder.parameters())
    out6.sum().backward()
    assert enc.grad is not None
