"""Tests for still-vs-orbit anomaly display helpers."""

from __future__ import annotations

import numpy as np

from tomography_ml_validation.plotting.anomaly_orbit_preview import (
    anomaly_camera_light_grid_html,
    anomaly_plane_to_rgb,
    anomaly_still_vs_orbit_html,
    build_anomaly_camera_light_grid_gif,
    build_anomaly_orbit_gif,
    build_anomaly_still_png,
)


def test_anomaly_plane_to_rgb_midgray_at_mean():
    plane = np.full((8, 8), 8.0e-5, dtype=np.float32)
    image = anomaly_plane_to_rgb(plane, mean=8.0e-5, std=1.0e-4, display_px=16)
    assert image.size == (16, 16)
    # Mid-gray channel ≈ 128 for value == mean.
    assert abs(image.getpixel((0, 0))[0] - 128) <= 1


def test_build_still_and_orbit_bytes():
    still = np.zeros((4, 4), dtype=np.float32)
    orbit = np.zeros((3, 4, 4), dtype=np.float32)
    orbit[1] = 1.0e-4
    png = build_anomaly_still_png(still, label="still")
    gif = build_anomaly_orbit_gif(
        orbit,
        sequence_id="seq",
        angles_deg=(0.0, 120.0, 240.0),
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert gif[:6] in (b"GIF87a", b"GIF89a")


def test_anomaly_still_vs_orbit_html_contains_both_media():
    still = np.zeros((4, 4), dtype=np.float32)
    orbit = np.zeros((2, 4, 4), dtype=np.float32)
    html = anomaly_still_vs_orbit_html(
        still,
        orbit,
        still_sequence_id="s8",
        orbit_sequence_id="s9",
        still_angle_deg=0.0,
        orbit_angles_deg=(0.0, 180.0),
        display_px=32,
    )
    text = str(html.data)
    assert "data:image/png;base64," in text
    assert "data:image/gif;base64," in text
    assert "Display only" in text


def test_anomaly_camera_light_grid_html():
    still = np.zeros((4, 4), dtype=np.float32)
    # Canonical [I, V, C, H, W]
    grid = np.zeros((3, 2, 1, 4, 4), dtype=np.float32)
    gif = build_anomaly_camera_light_grid_gif(
        grid,
        sequence_id="bear_m10_000_000001",
        camera_angles_deg=(0.0, 120.0),
        light_angles_deg=(0.0, 120.0, 240.0),
        fps=2.0,
        display_px=32,
    )
    assert gif[:6] in (b"GIF87a", b"GIF89a")

    html = anomaly_camera_light_grid_html(
        still,
        grid,
        sequence_id="bear_m10_000_000001",
        still_camera_deg=0.0,
        still_light_deg=0.0,
        camera_angles_deg=(0.0, 120.0),
        light_angles_deg=(0.0, 120.0, 240.0),
        fps=2.0,
        display_px=32,
    )
    text = str(html.data)
    assert "[3,2,1,4,4]" in text
    assert "[I,V,C,H,W]" in text
    assert text.count("data:image/gif;base64,") == 1
    assert text.count("data:image/png;base64,") == 1
    assert "2 fps" in text

    gif_only = anomaly_camera_light_grid_html(
        None,
        grid,
        sequence_id="bear_m10_000_000001",
        camera_angles_deg=(0.0, 120.0),
        light_angles_deg=(0.0, 120.0, 240.0),
        fps=2.0,
        display_px=32,
        include_still=False,
    )
    gif_text = str(gif_only.data)
    assert gif_text.count("data:image/gif;base64,") == 1
    assert "data:image/png;base64," not in gif_text
    assert "Still [" not in gif_text
