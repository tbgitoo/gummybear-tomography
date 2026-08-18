"""Figure 3 history / POV helpers (no full-corpus training required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from gummybear_illustration.figure3_export import (
    assemble_gif,
    figure3_layout,
    figure3_render_worker_count,
    write_figure3_pov_scenes,
)
from gummybear_illustration.figure3_history import (
    MODEL_FOURIER,
    MODEL_POOLING,
    append_record,
    combine_histories,
    default_figure3_root,
    empty_history,
    load_history,
    save_history,
    select_best_fourier_advantage_sample,
)
from gummybear_illustration.figure3_pov_scene import (
    approach_weight,
    build_figure3_epoch_pov,
    distance_luminosity,
    distance_transmit,
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
            train_loss=None if epoch == 0 else 2.0,
            val_loss=2.5,
        )
        append_record(
            fourier,
            epoch=epoch,
            sample_id="bear_a",
            y_true=[1.0, 2.0, 5.0],
            y_pred=[1.0 + 0.8 * (1 - epoch / 3), 2.0, 5.0],
            localization_error=0.8 * (1 - epoch / 3) + 0.05,
            train_loss=None if epoch == 0 else 0.5,
            val_loss=0.6,
        )
        append_record(
            pooling,
            epoch=epoch,
            sample_id="bear_b",
            y_true=[0.0, 0.0, 4.0],
            y_pred=[0.2, 0.2, 4.2],
            localization_error=0.35,
            train_loss=None if epoch == 0 else 2.0,
            val_loss=2.5,
        )
        append_record(
            fourier,
            epoch=epoch,
            sample_id="bear_b",
            y_true=[0.0, 0.0, 4.0],
            y_pred=[0.15, 0.15, 4.1],
            localization_error=0.25,
            train_loss=None if epoch == 0 else 0.5,
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
    text = povs[-1].read_text(encoding="utf-8")
    assert "Figure 3" in text
    assert "identity" in text.lower() or "simulation millimetres" in text
    assert "BearMesh" in text
    assert "sample bear_a" in text
    assert "sample bear_b" in text
    assert "n_val=2" in text
    assert "emission rgb" in text
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
