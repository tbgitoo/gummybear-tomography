"""Figure 3 history / POV helpers (no full-corpus training required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from gummybear_illustration.figure3_export import (
    add_bear_to_figure3_scenes,
    add_loss_board_to_figure3_scenes,
    add_markers_to_figure3_scenes,
    assemble_gif,
    composed_figure3_pov_text,
    figure3_layout,
    figure3_render_worker_count,
    generate_figure3_scenes,
    write_figure3_pov_scenes,
)
from gummybear_illustration.figure3_pov_scene import figure3_mesh_metrics
from gummybear_illustration.figure3_history import (
    MODEL_FOURIER,
    MODEL_POOLING,
    append_record,
    build_figure3_record_index,
    combine_histories,
    default_figure3_root,
    empty_history,
    load_history,
    records_for_sample,
    save_history,
    select_best_fourier_advantage_sample,
)
from gummybear_illustration.figure3_pov_scene import (
    approach_weight,
    build_figure3_epoch_pov,
    distance_luminosity,
    distance_transmit,
    figure3_apply_bear_placement,
    figure3_epoch_pairs,
    figure3_mesh_centroid,
    simulation_xyz_to_pov,
)


def _toy_histories() -> tuple[dict, dict, dict]:
    tracked = [
        {
            "sample_id": "bear_a",
            "sequence_id": "bear_a",
            "split": "validation",
            "y_true": [1.0, 2.0, 5.0],
        },
        {
            "sample_id": "bear_b",
            "sequence_id": "bear_b",
            "split": "validation",
            "y_true": [0.0, 0.0, 4.0],
        },
    ]
    pooling = empty_history(
        model_type=MODEL_POOLING,
        arch="pooled",
        lr=0.001,
        seed=0,
        num_epochs=3,
        y_fields=("particle_x", "particle_y", "particle_z"),
        tracked_samples=tracked,
    )
    fourier = empty_history(
        model_type=MODEL_FOURIER,
        arch="fourier",
        lr=0.03,
        seed=0,
        num_epochs=3,
        y_fields=("particle_x", "particle_y", "particle_z"),
        tracked_samples=tracked,
    )
    for epoch in range(0, 4):
        # Fourier converges; pooling stays far on bear_a.
        append_record(
            pooling,
            epoch=epoch,
            sample_id="bear_a",
            y_true=[1.0, 2.0, 5.0],
            y_pred=[1.0 + 2.5, 2.0 - 1.5, 5.0 + 1.0 - 0.1 * epoch],
            localization_error=3.0 - 0.05 * epoch,
            train_loss=2.5 if epoch == 0 else 2.0,
            val_loss=2.5,
        )
        append_record(
            fourier,
            epoch=epoch,
            sample_id="bear_a",
            y_true=[1.0, 2.0, 5.0],
            y_pred=[1.0 + 0.8 * (1 - epoch / 3), 2.0, 5.0],
            localization_error=0.8 * (1 - epoch / 3) + 0.05,
            train_loss=0.9 if epoch == 0 else 0.5,
            val_loss=0.6,
        )
        append_record(
            pooling,
            epoch=epoch,
            sample_id="bear_b",
            y_true=[0.0, 0.0, 4.0],
            y_pred=[0.2, 0.2, 4.2],
            localization_error=0.35,
            train_loss=2.5 if epoch == 0 else 2.0,
            val_loss=2.5,
        )
        append_record(
            fourier,
            epoch=epoch,
            sample_id="bear_b",
            y_true=[0.0, 0.0, 4.0],
            y_pred=[0.15, 0.15, 4.1],
            localization_error=0.25,
            train_loss=0.9 if epoch == 0 else 0.5,
            val_loss=0.6,
        )
    combined = combine_histories(pooling, fourier)
    return pooling, fourier, combined


def test_figure3_history_roundtrip(tmp_path: Path):
    pooling, fourier, combined = _toy_histories()
    p = tmp_path / "m8_pooling_history.json"
    f = tmp_path / "fourier_pooling_history.json"
    c = tmp_path / "combined_prediction_history.json"
    save_history(p, pooling)
    save_history(f, fourier)
    save_history(c, combined)
    loaded = load_history(c)
    assert loaded["schema_version"] == 1
    assert len(loaded["records"]) == 16
    sid, p_err, f_err, adv = select_best_fourier_advantage_sample(loaded)
    assert sid == "bear_a"
    assert adv > 1.0
    assert p_err > f_err


def test_simulation_xyz_identity():
    xyz = np.array([1.5, -2.0, 3.25])
    assert np.allclose(simulation_xyz_to_pov(xyz), xyz)


def test_default_figure3_root_is_under_outputs(tmp_path: Path):
    root = default_figure3_root(tmp_path)
    assert root == tmp_path / "outputs" / "figure3_learning_convergence"


def test_figure3_pov_and_gif_from_json(tmp_path: Path):
    _pooling, _fourier, combined = _toy_histories()
    layout = figure3_layout(tmp_path)
    for key in ("root", "scenes", "renders", "final"):
        layout[key].mkdir(parents=True, exist_ok=True)
    save_history(layout["combined"], combined)
    # Tiny cube mesh as stand-in STL.
    stl = tmp_path / "cube.stl"
    mesh = trimesh.creation.box(extents=(10.0, 8.0, 12.0))
    mesh.export(stl)
    povs, camera = write_figure3_pov_scenes(
        combined,
        layout=layout,
        repo=tmp_path,
        sample_id="bear_a",
        epochs=[0, 1, 2, 3],
        stl_path=stl,
    )
    assert len(povs) == 4
    assert abs(camera["location"][0]) > abs(camera["location"][1])
    text = composed_figure3_pov_text(povs[-1])
    assert "Figure 3" in text
    assert "identity" in text.lower() or "simulation millimetres" in text
    assert "BearMesh" in text
    assert "sample bear_a" in text
    assert "sample bear_b" in text
    assert "n_val=2" in text
    assert "emission rgb" in text
    assert "Figure 3 learning-rate chalkboard" in text
    # Synthetic PNG frames → GIF (no POV-Ray required).
    from PIL import Image

    pngs = []
    for i, pov in enumerate(povs):
        png = layout["renders"] / f"{pov.stem}.png"
        Image.new("RGB", (64, 48), (20 + 10 * i, 30, 40)).save(png)
        pngs.append(png)
    gif = assemble_gif(pngs, layout["final_gif"], duration_ms=50)
    assert gif.is_file()
    # POV builder direct call — all tracked validation samples, face-on camera.
    scene = build_figure3_epoch_pov(
        combined_history=combined,
        epoch=3,
        bear_inc_name="figure3_bear.inc",
        mesh=mesh,
    )
    assert "0.58, 0.10, 0.12" in scene
    assert "0.14, 0.34, 0.78" in scene
    assert "0.14, 0.82, 0.28" in scene
    assert "sample bear_a" in scene and "sample bear_b" in scene
    assert "yaw_deg=80" in scene
    loc_line = next(ln for ln in scene.splitlines() if ln.strip().startswith("location <"))
    nums = [float(p.strip()) for p in loc_line.split("<", 1)[1].split(">", 1)[0].split(",")]
    dx, dy = nums[0], nums[1]  # cube centered at origin
    assert abs(dx) > abs(dy), loc_line  # network-style +X, not lateral −Y
    assert abs(dx) > 20.0, loc_line  # long standoff, not mesh-scale
    assert "looks_like" not in scene
    assert "fade_distance" not in scene
    assert "ball_lights=False" in scene
    assert "draw_prediction_links=False" in scene
    assert "distance_luminosity=False" in scene
    assert "distance_transparency=True" in scene
    assert "No target→pred cylinders" in scene
    assert "Predicted balls fade when far" in scene
    assert "ambient_light rgb <0.055" in scene
    assert "diffuse 0.72" in scene
    assert "specular 0.14" in scene
    assert "  hollow\n  double_illuminate\n" in scene
    # Fourier is closer than pooling on bear_a at epoch 3 → higher approach weight.
    a_comment = next(ln for ln in scene.splitlines() if "sample bear_a" in ln)
    wp = float(a_comment.split("w_pool=")[1].split()[0])
    wf = float(a_comment.split("w_fourier=")[1].split()[0])
    xmit_f = float(a_comment.split("xmit_fourier=")[1].split()[0])
    assert wf > wp
    assert wf == 1.0
    assert xmit_f == 0.0
    assert "color rgbt" in scene
    assert approach_weight(0.0) == 1.0
    assert approach_weight(0.5) == 1.0
    assert approach_weight(5.0) > approach_weight(10.0)
    assert approach_weight(10.0) == 0.0
    assert approach_weight(50.0) == 0.0
    assert distance_luminosity(0.0, at_target=2.0) == 2.0
    assert distance_luminosity(1.0, hold_mm=1.0, zero_mm=10.0, at_target=2.0) == 2.0
    assert distance_luminosity(5.5, hold_mm=1.0, zero_mm=10.0, at_target=2.0) == 1.0
    assert distance_luminosity(20.0, zero_mm=10.0, at_target=2.0) == 0.0
    assert distance_transmit(0.0) == 0.0
    assert distance_transmit(1.0, hold_mm=1.0, zero_mm=10.0) == 0.0
    assert distance_transmit(10.0, hold_mm=1.0, zero_mm=10.0) == 0.95
    assert distance_transmit(20.0) == 0.95
    assert abs(distance_transmit(5.5, hold_mm=1.0, zero_mm=10.0) - 0.475) < 1e-3
    assert "light_intensity=1" in scene
    bright = build_figure3_epoch_pov(
        combined_history=combined,
        epoch=3,
        bear_inc_name="figure3_bear.inc",
        mesh=mesh,
        light_intensity=2.0,
    )
    assert "light_intensity=2" in bright
    split_emission = build_figure3_epoch_pov(
        combined_history=combined,
        epoch=3,
        bear_inc_name="figure3_bear.inc",
        mesh=mesh,
        pooling_luminosity_at_target=0.15,
        fourier_luminosity_at_target=0.45,
    )
    assert "pooling_luminosity_at_target=0.15" in split_emission
    assert "fourier_luminosity_at_target=0.45" in split_emission
    lit = build_figure3_epoch_pov(
        combined_history=combined,
        epoch=3,
        bear_inc_name="figure3_bear.inc",
        mesh=mesh,
        ball_lights=True,
    )
    assert "looks_like" in lit
    assert "fade_distance" in lit
    assert "ball_lights=True" in lit
    assert "  spotlight\n" in lit
    assert "falloff 15" in lit
    assert "fade_power 1" in lit
    assert "ball_light_intensity=8" in lit
    assert "point_at" in lit
    omni = build_figure3_epoch_pov(
        combined_history=combined,
        epoch=3,
        bear_inc_name="figure3_bear.inc",
        mesh=mesh,
        ball_lights=True,
        ball_spotlights=False,
    )
    assert "  spotlight\n" not in omni
    tight = build_figure3_epoch_pov(
        combined_history=combined,
        epoch=3,
        bear_inc_name="figure3_bear.inc",
        mesh=mesh,
        ball_lights=True,
        ball_spotlight_angle_deg=12.0,
    )
    assert "falloff 12" in tight
    a_lit = next(ln for ln in lit.splitlines() if "sample bear_a" in ln)
    b_lit = next(ln for ln in lit.splitlines() if "sample bear_b" in ln)
    assert "light_pool=False" in a_lit
    assert "light_fourier=True" in a_lit
    assert "inside_fourier=True" in a_lit
    assert "green_spot_pool=False" in a_lit
    assert "green_spot_fourier=True" in a_lit
    assert "green_hollow=True" in a_lit
    assert "light_pool=True" in b_lit
    assert "light_fourier=True" in b_lit
    assert "inside_pool=False" in b_lit
    assert "inside_fourier=True" in b_lit
    assert "green_spot_pool=False" in b_lit
    assert "green_spot_fourier=True" in b_lit
    assert "green_hollow=True" in b_lit
    # Green targets never get looks_like; only close predicted balls do.
    # Hits add a companion green light_source without a second sphere.
    assert lit.count("looks_like") == 3
    assert lit.count("  spotlight\n") == 5  # 3 pred colours + 2 green companions
    a_block = lit.split("sample bear_a")[1].split("sample bear_b")[0]
    assert "no_shadow hollow" in a_block
    assert "2.736" in a_block  # companion green TARGET_LIGHT * LIGHT_I_MAX * scale
    assert "3.456" in a_block  # Fourier blue light kept on the hit
    far = build_figure3_epoch_pov(
        combined_history=combined,
        epoch=0,
        bear_inc_name="figure3_bear.inc",
        mesh=mesh,
        ball_lights=True,
        pred_luminosity_hold_mm=0.05,
    )
    assert "looks_like" not in far
    linked = build_figure3_epoch_pov(
        combined_history=combined,
        epoch=3,
        bear_inc_name="figure3_bear.inc",
        mesh=mesh,
        draw_prediction_links=True,
    )
    assert "Target→pred cylinders drawn" in linked
    assert "rgbt <0.58, 0.10, 0.12, 0.45>" in linked
    opaque = build_figure3_epoch_pov(
        combined_history=combined,
        epoch=3,
        bear_inc_name="figure3_bear.inc",
        mesh=mesh,
        distance_transparency=False,
    )
    assert "Predicted balls opaque" in opaque
    assert "color rgbt" not in opaque.split("sample bear_a")[1].split("sample bear_b")[0]

    overridden = build_figure3_epoch_pov(
        combined_history=combined,
        epoch=3,
        bear_inc_name="figure3_bear.inc",
        mesh=mesh,
        camera_location=(40.0, -8.0, 12.0),
        camera_look_at=(1.0, 2.0, 3.0),
    )
    assert "location <40, -8, 12>" in overridden
    assert "look_at <1, 2, 3>" in overridden


def test_figure3_apply_bear_placement_identity():
    mesh = trimesh.creation.box(extents=(10.0, 8.0, 12.0))
    raw = np.array([1.5, -2.0, 3.0])
    placed = figure3_apply_bear_placement(raw, mesh=mesh, scale=1.0, translate=None)
    assert np.allclose(placed, raw)


def test_figure3_apply_bear_placement_scaled():
    mesh = trimesh.creation.box(extents=(10.0, 8.0, 12.0))
    center = figure3_mesh_centroid(mesh)
    raw = np.array([1.5, -2.0, 3.0])
    offset = np.array([-20.0, 0.0, 0.0])
    placed = figure3_apply_bear_placement(
        raw, mesh=mesh, scale=0.5, translate=offset
    )
    assert np.allclose(placed, offset + 0.5 * (raw - center))


def test_add_loss_board_to_figure3_scenes(tmp_path: Path):
    _pooling, _fourier, combined = _toy_histories()
    layout = figure3_layout(tmp_path)
    for key in ("root", "scenes", "renders", "final"):
        layout[key].mkdir(parents=True, exist_ok=True)
    stl = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(10.0, 8.0, 12.0)).export(stl)
    scenes = generate_figure3_scenes(
        layout=layout,
        repo=tmp_path,
        epochs=[0, 2],
        combined=combined,
        stl_path=stl,
    )
    add_loss_board_to_figure3_scenes(
        scenes,
        combined,
        curve_width_px=64,
        dot_radius_mm=1.0,
    )
    stub = scenes.pov_paths[0].read_text(encoding="utf-8")
    assert "board_loss.inc" in stub
    object_inc = (scenes.scenes_dir / "figure3_board_object.inc").read_text(
        encoding="utf-8"
    )
    dot0 = (scenes.scenes_dir / "figure3_epoch_0000_board_loss.inc").read_text(
        encoding="utf-8"
    )
    dot_inc = (scenes.scenes_dir / "figure3_epoch_0002_board_loss.inc").read_text(
        encoding="utf-8"
    )
    assert "figure3_board_object.inc" in stub
    assert "Figure 3 learning-rate chalkboard" in object_inc
    assert 'image_map { png "figure3_loss_board_pngs/figure3_loss_board_curves.png"' in object_inc
    assert object_inc.count("matrix <") == 2
    assert "mesh2" not in object_inc
    assert "figure3_loss_board_curves.png" not in dot_inc
    assert "Figure 3 learning-rate board dots" in dot0
    assert "omitted" not in dot0
    assert "light_source {" in dot0
    assert "Figure 3 learning-rate board dots" in dot_inc
    assert "light_source {" in dot_inc
    assert "sphere { 0, 1" in dot_inc
    assert (scenes.scenes_dir / "figure3_loss_board_pngs" / "figure3_loss_board_curves.png").is_file()
    from gummybear_illustration.figure3_export import (
        FIGURE3_BOARD_TITLE,
        _draw_figure3_loss_board_image,
        _figure3_train_loss_by_epoch,
        _loss_at_or_before_epoch,
    )

    series = _figure3_train_loss_by_epoch(combined, model_type=MODEL_POOLING)
    assert series[0][0] == 0
    assert _loss_at_or_before_epoch(series, 0) is not None

    img = _draw_figure3_loss_board_image(width=640, height=440, combined=combined)
    # Title is baked into the opaque RGB texture; sample a few chalk-ink pixels
    # near the top center where "Training loss" is drawn.
    arr = np.asarray(img.convert("RGB"))
    band = arr[8 : max(9, arr.shape[0] // 5), arr.shape[1] // 4 : 3 * arr.shape[1] // 4]
    assert FIGURE3_BOARD_TITLE == "Training loss"
    assert band[..., 0].max() > 180  # chalk ink present in the title band


def test_figure3_train_loss_epoch0_from_preds_fallback():
    """Older histories omit epoch-0 ``train_loss``; synthesize from init preds."""
    from gummybear_illustration.figure3_export import _figure3_train_loss_by_epoch

    combined = {
        "num_epochs": 2,
        "records": [
            {
                "epoch": 0,
                "model_type": MODEL_POOLING,
                "sample_id": "a",
                "y_true": [0.0, 0.0, 0.0],
                "y_pred": [2.0, 0.0, 0.0],
                "train_loss": None,
            },
            {
                "epoch": 0,
                "model_type": MODEL_POOLING,
                "sample_id": "b",
                "y_true": [0.0, 0.0, 0.0],
                "y_pred": [0.0, 2.0, 0.0],
                "train_loss": None,
            },
            {
                "epoch": 1,
                "model_type": MODEL_POOLING,
                "sample_id": "a",
                "y_true": [0.0, 0.0, 0.0],
                "y_pred": [1.0, 0.0, 0.0],
                "train_loss": 1.0,
            },
        ],
    }
    series = _figure3_train_loss_by_epoch(combined, model_type=MODEL_POOLING)
    assert series[0][0] == 0
    assert abs(series[0][1] - (4.0 / 3.0)) < 1e-9
    assert series[1] == (1, 1.0)


def test_annotate_figure3_bear_labels(tmp_path: Path):
    """Post-render captions sit under Training / Validation / Test bears."""
    from PIL import Image as _Image

    from gummybear_illustration.figure3_export import (
        annotate_figure3_bear_labels,
        figure3_bear_label_pixel_anchors,
    )

    _pooling, _fourier, combined = _toy_histories()
    layout = figure3_layout(tmp_path)
    for key in ("root", "scenes", "renders", "final"):
        layout[key].mkdir(parents=True, exist_ok=True)
    stl = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(10.0, 8.0, 12.0)).export(stl)
    scenes = generate_figure3_scenes(
        layout=layout,
        repo=tmp_path,
        epochs=[0],
        combined=combined,
        stl_path=stl,
        camera_location=(-15.0, 85.0, 8.0),
        camera_look_at=(-15.1, 0.5, 10.5),
        fov_deg=42.0,
    )
    _, _, span = figure3_mesh_metrics(scenes.mesh)
    add_bear_to_figure3_scenes(scenes, bear_id="default")
    add_bear_to_figure3_scenes(
        scenes,
        bear_id="validation",
        scale=0.38,
        translate=tuple(s * span for s in (-1.5, 1.5, 0.25)),
    )
    add_bear_to_figure3_scenes(
        scenes,
        bear_id="train",
        scale=0.38,
        translate=tuple(s * span for s in (-2.3, 1.5, 0.25)),
    )
    anchors = figure3_bear_label_pixel_anchors(scenes, width=960, height=720)
    captions = [c for c, _x, _y in anchors]
    assert captions == ["Training", "Validation", "Test"]
    assert anchors[0][1] < anchors[1][1] < anchors[2][1]

    png = tmp_path / "frame.png"
    _Image.new("RGB", (960, 720), (20, 24, 30)).save(png)
    annotate_figure3_bear_labels([png], scenes, font_px=28)
    arr = np.asarray(_Image.open(png).convert("RGB"))
    for _caption, px, py in anchors:
        x = int(round(px))
        y = int(round(py + 0.012 * 720))
        patch = arr[
            max(0, y - 6) : min(720, y + 36),
            max(0, x - 60) : min(960, x + 60),
        ]
        assert patch[..., 0].max() > 180


def test_figure3_loss_board_right_frac_sign():
    """Negative ``center_right_frac`` shifts the board to the viewer's left."""
    from gummybear_illustration.figure3_export import _camera_basis

    loc = np.array([-15.0, 85.0, 8.0])
    look = np.array([-15.1, 0.5, 10.5])
    dist = float(np.linalg.norm(look - loc))
    _fwd, right, _up = _camera_basis(loc, look)
    frac = -0.33
    board_center = loc + frac * dist * right
    assert board_center[0] < loc[0]
    assert np.dot(board_center - loc, right) < 0.0


def test_figure3_render_worker_count(monkeypatch):
    monkeypatch.setattr(
        "gummybear_illustration.figure3_export.default_pov_render_workers",
        lambda: 6,
    )
    assert figure3_render_worker_count(20, multiprocessing=True) == 6
    assert figure3_render_worker_count(3, multiprocessing=True) == 3
    assert figure3_render_worker_count(1, multiprocessing=True) == 1
    assert figure3_render_worker_count(20, multiprocessing=False) == 1
    assert figure3_render_worker_count(0, multiprocessing=True) == 1


def test_figure3_scenes_are_layered(tmp_path: Path):
    _pooling, _fourier, combined = _toy_histories()
    layout = figure3_layout(tmp_path)
    for key in ("root", "scenes", "renders", "final"):
        layout[key].mkdir(parents=True, exist_ok=True)
    stl = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(10.0, 8.0, 12.0)).export(stl)
    scenes = generate_figure3_scenes(
        layout=layout,
        repo=tmp_path,
        epochs=[0, 3],
        combined=combined,
        stl_path=stl,
        sample_id="bear_a",
    )
    stub = scenes.pov_paths[-1].read_text(encoding="utf-8")
    assert '#include "figure3_world.inc"' in stub
    assert "figure3_bear_object.inc" not in stub
    world = composed_figure3_pov_text(scenes.pov_paths[-1])
    assert "double_illuminate" not in world
    assert "sample bear_a" not in world
    add_bear_to_figure3_scenes(scenes)
    stub = scenes.pov_paths[-1].read_text(encoding="utf-8")
    assert '#include "figure3_bear.inc"' in stub
    assert '#include "figure3_bear_object.inc"' in stub
    with_bear = composed_figure3_pov_text(scenes.pov_paths[-1])
    assert "  hollow\n  double_illuminate\n" in with_bear
    assert "sample bear_a" not in with_bear
    add_markers_to_figure3_scenes(scenes, combined, ball_lights=True)
    with_markers = composed_figure3_pov_text(scenes.pov_paths[-1])
    assert "sample bear_a" in with_markers
    assert "sample bear_b" in with_markers
    assert "looks_like" in with_markers
    stub = scenes.pov_paths[-1].read_text(encoding="utf-8")
    assert "markers_default_validation.inc" in stub


def test_figure3_markers_split_and_bear(tmp_path: Path):
    _pooling, _fourier, combined = _toy_histories()
    combined["tracked_samples"].append(
        {
            "sample_id": "bear_train",
            "sequence_id": "bear_train",
            "split": "train",
            "y_true": [2.0, 0.0, 4.0],
        }
    )
    for epoch in range(0, 4):
        combined["records"].extend(
            [
                {
                    "model_type": MODEL_POOLING,
                    "epoch": epoch,
                    "sample_id": "bear_train",
                    "y_true": [2.0, 0.0, 4.0],
                    "y_pred": [3.0, 0.0, 4.0],
                    "localization_error": 1.0,
                    "train_loss": None,
                    "val_loss": None,
                },
                {
                    "model_type": MODEL_FOURIER,
                    "epoch": epoch,
                    "sample_id": "bear_train",
                    "y_true": [2.0, 0.0, 4.0],
                    "y_pred": [2.1, 0.0, 4.0],
                    "localization_error": 0.1,
                    "train_loss": None,
                    "val_loss": None,
                },
            ]
        )
    layout = figure3_layout(tmp_path)
    for key in ("root", "scenes", "renders", "final"):
        layout[key].mkdir(parents=True, exist_ok=True)
    stl = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(10.0, 8.0, 12.0)).export(stl)
    scenes = generate_figure3_scenes(
        layout=layout,
        repo=tmp_path,
        epochs=[3],
        combined=combined,
        stl_path=stl,
    )
    add_bear_to_figure3_scenes(scenes, bear_id="default")
    add_markers_to_figure3_scenes(
        scenes, combined, split="validation", bear_id="default"
    )
    add_markers_to_figure3_scenes(
        scenes, combined, split="train", bear_id="default"
    )
    stub = scenes.pov_paths[-1].read_text(encoding="utf-8")
    assert "markers_default_validation.inc" in stub
    assert "markers_default_train.inc" in stub
    assert [layer["split"] for layer in scenes.marker_layers] == [
        "validation",
        "train",
    ]
    val_inc = scenes.scenes_dir / "figure3_epoch_0003_markers_default_validation.inc"
    train_inc = scenes.scenes_dir / "figure3_epoch_0003_markers_default_train.inc"
    val_text = val_inc.read_text(encoding="utf-8")
    train_text = train_inc.read_text(encoding="utf-8")
    assert "sample bear_a" in val_text
    assert "sample bear_train" not in val_text
    assert "sample bear_train" in train_text
    assert "sample bear_a" not in train_text
    assert "bear_id=default split=validation" in val_text
    assert "bear_id=default split=train" in train_text
    try:
        add_markers_to_figure3_scenes(
            scenes, combined, split="validation", bear_id="other_bear"
        )
        raise AssertionError("expected missing bear")
    except ValueError as exc:
        assert "other_bear" in str(exc)


def test_figure3_bear_placement_and_test_split(tmp_path: Path):
    _pooling, _fourier, combined = _toy_histories()
    combined["tracked_samples"].extend(
        [
            {
                "sample_id": "bear_test",
                "sequence_id": "bear_test",
                "split": "test",
                "y_true": [-1.0, 1.0, 4.0],
            },
            {
                "sample_id": "bear_train",
                "sequence_id": "bear_train",
                "split": "train",
                "y_true": [2.0, 0.0, 4.0],
            },
        ]
    )
    for epoch in range(0, 4):
        combined["records"].extend(
            [
                {
                    "model_type": MODEL_POOLING,
                    "epoch": epoch,
                    "sample_id": "bear_test",
                    "y_true": [-1.0, 1.0, 4.0],
                    "y_pred": [0.0, 1.0, 4.0],
                    "localization_error": 1.0,
                    "train_loss": None,
                    "val_loss": None,
                },
                {
                    "model_type": MODEL_FOURIER,
                    "epoch": epoch,
                    "sample_id": "bear_test",
                    "y_true": [-1.0, 1.0, 4.0],
                    "y_pred": [-0.9, 1.0, 4.0],
                    "localization_error": 0.1,
                    "train_loss": None,
                    "val_loss": None,
                },
                {
                    "model_type": MODEL_POOLING,
                    "epoch": epoch,
                    "sample_id": "bear_train",
                    "y_true": [2.0, 0.0, 4.0],
                    "y_pred": [3.0, 0.0, 4.0],
                    "localization_error": 1.0,
                    "train_loss": None,
                    "val_loss": None,
                },
                {
                    "model_type": MODEL_FOURIER,
                    "epoch": epoch,
                    "sample_id": "bear_train",
                    "y_true": [2.0, 0.0, 4.0],
                    "y_pred": [2.1, 0.0, 4.0],
                    "localization_error": 0.1,
                    "train_loss": None,
                    "val_loss": None,
                },
            ]
        )
    layout = figure3_layout(tmp_path)
    for key in ("root", "scenes", "renders", "final"):
        layout[key].mkdir(parents=True, exist_ok=True)
    stl = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(10.0, 8.0, 12.0)).export(stl)
    scenes = generate_figure3_scenes(
        layout=layout,
        repo=tmp_path,
        epochs=[3],
        combined=combined,
        stl_path=stl,
    )
    add_bear_to_figure3_scenes(scenes, bear_id="default")
    add_bear_to_figure3_scenes(
        scenes, bear_id="validation", scale=0.5, translate=(-20.0, 0.0, 5.0)
    )
    add_bear_to_figure3_scenes(
        scenes, bear_id="train", scale=0.38, translate=(-20.0, 12.0, 4.0)
    )
    add_markers_to_figure3_scenes(
        scenes, combined, split="test", bear_id="default"
    )
    add_markers_to_figure3_scenes(
        scenes, combined, split="validation", bear_id="validation"
    )
    add_markers_to_figure3_scenes(
        scenes, combined, split="train", bear_id="train"
    )
    stub = scenes.pov_paths[-1].read_text(encoding="utf-8")
    assert "markers_default_test.inc" in stub
    assert "markers_validation_validation.inc" in stub
    assert "markers_train_train.inc" in stub
    validation_object = (
        scenes.scenes_dir / "figure3_bear_validation_object.inc"
    ).read_text(encoding="utf-8")
    train_object = (scenes.scenes_dir / "figure3_bear_train_object.inc").read_text(
        encoding="utf-8"
    )
    assert "union {" in validation_object
    assert "scale 0.5" in validation_object
    assert "translate <-20, 0, 5>" in validation_object
    assert "scale 0.38" in train_object
    assert "translate <-20, 12, 4>" in train_object
    test_inc = scenes.scenes_dir / "figure3_epoch_0003_markers_default_test.inc"
    validation_val_inc = (
        scenes.scenes_dir / "figure3_epoch_0003_markers_validation_validation.inc"
    )
    train_inc = scenes.scenes_dir / "figure3_epoch_0003_markers_train_train.inc"
    assert "sample bear_test" in test_inc.read_text(encoding="utf-8")
    assert "sample bear_a" in validation_val_inc.read_text(encoding="utf-8")
    assert "sample bear_test" not in validation_val_inc.read_text(encoding="utf-8")
    assert "sample bear_train" in train_inc.read_text(encoding="utf-8")
    assert "sample bear_a" not in train_inc.read_text(encoding="utf-8")


def test_figure3_record_index_matches_linear_scan():
    _pooling, _fourier, combined = _toy_histories()
    index = build_figure3_record_index(combined)
    for sid in ("bear_a", "bear_b"):
        for model_type in (MODEL_POOLING, MODEL_FOURIER):
            assert records_for_sample(
                combined, sid, model_type=model_type, record_index=index
            ) == records_for_sample(combined, sid, model_type=model_type)
    for epoch in (0, 2, 3):
        assert figure3_epoch_pairs(
            combined,
            epoch,
            split="validation",
            record_index=index,
            split_sample_ids=["bear_a", "bear_b"],
        ) == figure3_epoch_pairs(combined, epoch, split="validation")


def test_figure3_marker_includes_parallel_matches_sequential(tmp_path: Path):
    _pooling, _fourier, combined = _toy_histories()
    layout = figure3_layout(tmp_path)
    for key in ("root", "scenes", "renders", "final"):
        layout[key].mkdir(parents=True, exist_ok=True)
    stl = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(10.0, 8.0, 12.0)).export(stl)

    def _build(*, parallel: bool) -> dict[str, str]:
        out_dir = tmp_path / ("parallel" if parallel else "sequential")
        out_layout = figure3_layout(out_dir)
        for key in ("root", "scenes", "renders", "final"):
            out_layout[key].mkdir(parents=True, exist_ok=True)
        scenes = generate_figure3_scenes(
            layout=out_layout,
            repo=tmp_path,
            epochs=[0, 1, 2, 3],
            combined=combined,
            stl_path=stl,
            sample_id="bear_a",
        )
        add_bear_to_figure3_scenes(scenes)
        add_markers_to_figure3_scenes(
            scenes,
            combined,
            ball_lights=True,
            multiprocessing=parallel,
        )
        marker_dir = scenes.scenes_dir
        return {
            path.name: path.read_text(encoding="utf-8")
            for path in sorted(marker_dir.glob("figure3_epoch_*_markers_*.inc"))
        }

    assert _build(parallel=False) == _build(parallel=True)
