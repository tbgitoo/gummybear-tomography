"""POV-Ray network-inference illustration (no torch/POV required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from gummybear_illustration.export_m8_network_scene import export_m8_network_scene
from gummybear_illustration.network_activations import fallback_activation_bundle
from gummybear_illustration.network_captions import overlay_network_captions
from gummybear_illustration.network_pov_scene import pipeline_to_world


def test_export_network_scene_fallback(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    z = np.zeros((128, 128), dtype=np.float32)
    z[20:90, 25:100] = 1.8
    bundle = fallback_activation_bundle(z, y_true=np.array([2.0, 1.0, 5.0]))
    out = tmp_path / "outputs" / "pov" / "net.pov"
    result = export_m8_network_scene(
        output_pov=out,
        activations=bundle,
        repo_root_path=tmp_path,
        render=False,
    )
    text = result["pov"].read_text(encoding="utf-8")
    assert "#version 3.7" in text
    assert "per-channel" in text
    assert (out.parent / "net_c16_xm.png").is_file()
    assert (out.parent / "net_c16_yp.png").is_file()
    assert "Prediction bear:" in text
    assert "Target bear (50% scale)" in text
    assert "BearMesh" in text
    assert "emission rgb <0.45, 0.72, 2.4>" in text
    assert "emission rgb <2.35, 0.38, 0.28>" in text
    assert "emission rgb <0.45, 1.7, 0.55>" in text
    assert "Fourier-path xyz prediction" in text
    assert "GAP/pooling-path xyz prediction" in text
    assert "Ground-truth particle centre" in text
    assert "3D localization panel" not in text
    assert "transmit=0.88" in text
    assert text.count("rgb <0.14, 0.38, 0.62>") >= 4
    assert text.count("rgb <0.52, 0.08, 0.10>") >= 6
    assert text.count("rgb <0.16, 0.34, 0.82>") >= 2
    assert "Readout arrows:" in text
    assert "Fourier term" in text
    assert (out.parent / "net_fourier_pane.png").is_file()
    assert (out.parent / "net_gap_pane.png").is_file()
    assert "GAP-branch last-layer maps" in text
    assert "Fourier pooled embedding" in text
    assert (out.parent / "net_fourier_pool.png").is_file()
    assert text.count("net_gap_pool.png") == 6
    assert text.count("net_fourier_pool.png") == 6
    assert "<0,0>, <0,1>, <0,1>, <0,0>" not in text
    assert "<1,0>, <1,1>, <1,1>, <1,0>" not in text
    assert "reserved for later representations" not in text
    assert "Activation source: fallback" in text
    assert "interpolate 0" not in text
    assert "single-view" in text
    assert text.count("_input_zscore.png") == 2
    assert "both faces" in text
    assert text.count("rotate <0, 0, -90>") == 4
    assert (out.parent / "net_gap_c16_xm.png").is_file()
    assert (out.parent / "net_gap_c64_yp.png").is_file()
    assert "GAP-branch CNN volumes" in text
    assert text.count("net_gap_c16_xm.png") >= 1
    assert np.allclose(
        pipeline_to_world(np.array([2.0, 0.0, 8.0])),
        np.array([48.0, 46.0, 8.0]),
    )
    assert "single-view or multi-view input" not in text
    assert (out.parent / "net_c16_xm.png").is_file()
    assert (out.parent / "net_c16_yp.png").is_file()
    assert (out.parent / "net_fourier_00.png").is_file()
    assert "png" not in result


def test_fourier_embed_offset_moves_slab(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    z = np.zeros((128, 128), dtype=np.float32)
    bundle = fallback_activation_bundle(z, y_true=np.array([2.0, 1.0, 5.0]))
    kwargs = dict(activations=bundle, repo_root_path=tmp_path, render=False)
    a = export_m8_network_scene(
        output_pov=tmp_path / "outputs" / "pov" / "a.pov",
        fourier_embed_offset=(0.28, 2.6, 0.0),
        **kwargs,
    )["pov"].read_text(encoding="utf-8")
    b = export_m8_network_scene(
        output_pov=tmp_path / "outputs" / "pov" / "b.pov",
        fourier_embed_offset=(4.0, 12.0, 3.5),
        **kwargs,
    )["pov"].read_text(encoding="utf-8")
    c = export_m8_network_scene(
        output_pov=tmp_path / "outputs" / "pov" / "c.pov",
        fourier_embed_scale=0.4,
        **kwargs,
    )["pov"].read_text(encoding="utf-8")
    d = export_m8_network_scene(
        output_pov=tmp_path / "outputs" / "pov" / "d.pov",
        gap_embed_yaw_deg=35.0,
        gap_embed_scale=0.6,
        gap_embed_shift=8.0,
        **kwargs,
    )["pov"].read_text(encoding="utf-8")
    assert a != b
    assert a != c
    assert a != d
    assert "just in front of the Fourier pane" in a
    assert "just in front of the Fourier pane" in b


def test_overlay_network_captions(tmp_path: Path):
    path = tmp_path / "n.png"
    Image.new("RGB", (1280, 960), (255, 255, 255)).save(path)
    overlay_network_captions(path, positions={"cnn": (0.2, 0.3)})
    pixels = np.asarray(Image.open(path))
    dark = pixels.min(axis=2) < 40
    y, x = int(0.3 * 960), int(0.2 * 1280)
    assert dark[max(0, y - 40) : y + 40, max(0, x - 40) : x + 40].any()


def test_overlay_network_caption_branch_colors(tmp_path: Path):
    path = tmp_path / "c.png"
    Image.new("RGB", (400, 300), (255, 255, 255)).save(path)
    hide = {
        k: None
        for k in (
            "input",
            "cnn",
            "layer1",
            "layer2",
            "layer3",
            "layer1_gap",
            "layer2_gap",
            "layer3_gap",
            "cnn_readout",
            "embedding",
            "embedding_gap",
            "mlp",
            "pred_gap",
            "pred_fourier",
        )
    }
    overlay_network_captions(
        path,
        positions={
            **hide,
            "gap_branch": (0.25, 0.25),
            "fourier_branch": (0.75, 0.25),
            "target": (0.50, 0.75),
        },
    )
    pixels = np.asarray(Image.open(path), dtype=int)
    y_g, x_g = int(0.25 * 300), int(0.25 * 400)
    y_f, x_f = int(0.25 * 300), int(0.75 * 400)
    y_t, x_t = int(0.75 * 300), int(0.50 * 400)
    gap = pixels[max(0, y_g - 24) : y_g + 6, max(0, x_g - 50) : x_g + 50]
    fou = pixels[max(0, y_f - 24) : y_f + 6, max(0, x_f - 50) : x_f + 50]
    tgt = pixels[max(0, y_t - 24) : y_t + 6, max(0, x_t - 50) : x_t + 50]
    assert gap[..., 0].min() < 200 and gap[..., 0].mean() > gap[..., 2].mean()
    assert fou[..., 2].min() < 200 and fou[..., 2].mean() > fou[..., 0].mean()
    assert tgt[..., 1].min() < 200 and tgt[..., 1].mean() > tgt[..., 0].mean()


def test_fourier_prepool_matches_pooled():
    z = np.zeros((128, 128), dtype=np.float32)
    z[20:90, 25:100] = 1.8
    bundle = fallback_activation_bundle(z, y_true=np.array([2.0, 1.0, 5.0]))
    assert bundle.fourier_prepool.shape == (64, 128, 128)
    assert np.allclose(bundle.fourier_prepool.mean(axis=(1, 2)), bundle.fourier_pooled)
    assert np.allclose(bundle.gap_prepool.mean(axis=(1, 2)), bundle.gap)
    assert bundle.flatten_vec.shape == (64 * 128 * 128,)
    assert np.allclose(bundle.flatten_vec, bundle.flatten_pre.reshape(-1))
    assert bundle.y_pred_flatten.shape == (3,)


def test_channel_colormap_matches_imshow_low_background():
    """Hotspot planes stay turbo-blue on the bulk, not percentile-stretched green."""
    from matplotlib import colormaps

    from gummybear_illustration.network_textures import (
        _to_uint8_rgb,
        channel_colormap_limits,
    )

    vol = np.zeros((1, 16, 16), dtype=float)
    vol[0, 2:5, 2:5] = 10.0
    vmin, vmax = channel_colormap_limits(vol)
    rgb = _to_uint8_rgb(vol[0, 0, :], vmin=float(vmin[0]), vmax=float(vmax[0]))
    turbo0 = (np.asarray(colormaps["turbo"](0.0)[:3]) * 255).astype(np.uint8)
    turbo_mid = (np.asarray(colormaps["turbo"](0.5)[:3]) * 255).astype(np.uint8)
    assert np.mean(np.abs(rgb.astype(int) - turbo0)) < np.mean(
        np.abs(rgb.astype(int) - turbo_mid)
    )


def test_channel_colormap_limits_independent():
    from gummybear_illustration.network_textures import channel_colormap_limits

    vol = np.zeros((2, 8, 8), dtype=float)
    vol[0] = 1.0
    vol[1] = 80.0
    vmin, vmax = channel_colormap_limits(vol)
    assert vmax[0] < vmin[1]


def test_volume_boundary_faces_share_edges():
    from gummybear_illustration.network_textures import volume_boundary_faces

    vol = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    faces = volume_boundary_faces(vol)
    assert np.allclose(faces["xm"][:, 0], faces["ym"][:, 0])
    assert np.allclose(faces["xm"][0, :], faces["zm"][:, 0])
    assert np.allclose(faces["xp"][:, -1], faces["yp"][:, -1])
    assert np.allclose(faces["xp"][-1, :], faces["zp"][:, -1])


def test_fourier_pane_top_row_swapped(tmp_path: Path):
    from gummybear_illustration.network_textures import write_fourier_pane_png

    arr = np.zeros((64, 8, 8), dtype=float)
    arr[0, 4, 4] = 1.0
    arr[9] = 1.0
    arr[9, 4, 4] = 0.0
    path = tmp_path / "pane.png"
    write_fourier_pane_png(arr, path)
    rgb = np.asarray(Image.open(path))
    gap, h, w = 10, 8, 8
    top_left = rgb[gap : gap + h, gap : gap + w].mean()
    top_right = rgb[gap : gap + h, 2 * gap + w : 2 * gap + 2 * w].mean()
    assert top_left > top_right


def test_pane_tiles_use_per_plane_scale_not_global(tmp_path: Path):
    from matplotlib import colormaps

    from gummybear_illustration.network_textures import write_fourier_pane_png

    arr = np.zeros((64, 8, 8), dtype=float)
    arr[0] = 40.0
    arr[9, 4, 4] = 1.0
    path = tmp_path / "pane.png"
    write_fourier_pane_png(arr, path)
    rgb = np.asarray(Image.open(path))
    gap, h, w = 10, 8, 8
    top_right = rgb[gap : gap + h, 2 * gap + w : 2 * gap + 2 * w]
    turbo0 = (np.asarray(colormaps["turbo"](0.0)[:3]) * 255).astype(np.uint8)
    turbo1 = (np.asarray(colormaps["turbo"](1.0)[:3]) * 255).astype(np.uint8)
    mean = top_right.mean(axis=(0, 1))
    assert float(top_right[..., 0].std()) < 1.0
    assert np.mean(np.abs(mean.astype(int) - turbo0)) < np.mean(
        np.abs(mean.astype(int) - turbo1)
    )


def test_slice_zscore_colormap_maps_zero_to_turbo_mid():
    from matplotlib import colormaps

    from gummybear_illustration.network_textures import _to_uint8_rgb

    rgb = _to_uint8_rgb(np.full((8, 8), 7.0), scale="zscore", zscore_clip=2.0)
    turbo_mid = (np.asarray(colormaps["turbo"](0.5)[:3]) * 255).astype(np.uint8)
    turbo0 = (np.asarray(colormaps["turbo"](0.0)[:3]) * 255).astype(np.uint8)
    mean = rgb.mean(axis=(0, 1))
    assert np.mean(np.abs(mean.astype(int) - turbo_mid)) < np.mean(
        np.abs(mean.astype(int) - turbo0)
    )


def test_fourier_group_offset_moves_pane_and_slab(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    z = np.zeros((128, 128), dtype=np.float32)
    bundle = fallback_activation_bundle(z, y_true=np.array([2.0, 1.0, 5.0]))
    kwargs = dict(activations=bundle, repo_root_path=tmp_path, render=False)
    a = export_m8_network_scene(
        output_pov=tmp_path / "outputs" / "pov" / "g0.pov",
        **kwargs,
    )["pov"].read_text(encoding="utf-8")
    b = export_m8_network_scene(
        output_pov=tmp_path / "outputs" / "pov" / "g1.pov",
        fourier_group_offset=(0.0, 8.0, 2.0),
        **kwargs,
    )["pov"].read_text(encoding="utf-8")
    assert a != b


def test_cnn_fourier_front_moves_volumes(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    z = np.zeros((128, 128), dtype=np.float32)
    bundle = fallback_activation_bundle(z, y_true=np.array([2.0, 1.0, 5.0]))
    kwargs = dict(activations=bundle, repo_root_path=tmp_path, render=False)
    a = export_m8_network_scene(
        output_pov=tmp_path / "outputs" / "pov" / "f0.pov",
        **kwargs,
    )["pov"].read_text(encoding="utf-8")
    b = export_m8_network_scene(
        output_pov=tmp_path / "outputs" / "pov" / "f1.pov",
        cnn_fourier_front=12.0,
        **kwargs,
    )["pov"].read_text(encoding="utf-8")
    assert a != b


def test_input_stack_scale_enlarges_plate(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    z = np.zeros((128, 128), dtype=np.float32)
    bundle = fallback_activation_bundle(z, y_true=np.array([2.0, 1.0, 5.0]))
    kwargs = dict(activations=bundle, repo_root_path=tmp_path, render=False)
    a = export_m8_network_scene(
        output_pov=tmp_path / "outputs" / "pov" / "s0.pov",
        **kwargs,
    )["pov"].read_text(encoding="utf-8")
    b = export_m8_network_scene(
        output_pov=tmp_path / "outputs" / "pov" / "s1.pov",
        input_stack_scale=1.5,
        **kwargs,
    )["pov"].read_text(encoding="utf-8")
    assert a != b
