"""Tests for M8 WIN 0E dataset validation panels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gummybear_validation.milestone_08 import project_world_points_to_pixels


def test_project_look_at_lands_near_image_center():
    camera_position = (0.0, -80.0, 0.0)
    look_at = (0.0, 0.0, 0.0)
    up = (0.0, 0.0, 1.0)
    uv, in_front = project_world_points_to_pixels(
        np.asarray(look_at),
        camera_position=camera_position,
        look_at=look_at,
        up=up,
        fov_deg=35.0,
        resolution=(128, 128),
    )
    assert bool(in_front[0])
    np.testing.assert_allclose(uv[0], (63.5, 63.5), atol=0.5)


def test_regime_histogram_xlim_uses_quantile_not_max(monkeypatch):
    """X-axis should follow a high quantile so rare hot pixels do not dominate."""
    import matplotlib

    matplotlib.use("Agg")

    from gummybear_validation.milestone_08 import plot_dataset_panels as panels
    from gummybear_validation.milestone_08 import plot_regime_intensity_histograms

    @dataclass(frozen=True)
    class _FakeRef:
        name: str

    @dataclass(frozen=True)
    class _FakeRow:
        sequence_id: str
        bear_mu_a: float
        bear_mu_s: float
        observed_ref: _FakeRef

    arrays = {
        "low": np.array([0.0, 0.0, 0.001, 0.002], dtype=np.float32).reshape(
            1, 1, 2, 2
        ),
        "high": np.array([0.0, 0.01, 0.01, 1.0], dtype=np.float32).reshape(1, 1, 2, 2),
    }

    def _fake_load(ref, **kwargs):
        return arrays[ref.name]

    monkeypatch.setattr(panels, "load_role_array", _fake_load)

    rows = {
        "low": _FakeRow("bear_low", 0.01, 0.03, _FakeRef("low")),
        "high": _FakeRow("bear_high", 0.1, 0.3, _FakeRef("high")),
    }
    fig = plot_regime_intensity_histograms(
        rows,  # type: ignore[arg-type]
        bins=16,
        xlim_quantile=0.75,
        xlim_pad=0.0,
    )
    axis = fig.axes[0]
    left, right = axis.get_xlim()
    assert left > 0.0  # zeros excluded; axis starts at positive min
    # Pooled positive values include a lone 1.0 max; q75 should keep xmax below that.
    assert right < 0.5
    assert "zeros excluded" in axis.get_xlabel()
    assert "xlim=p75" in axis.get_xlabel()
