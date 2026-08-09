"""Smoke tests for M8 localisation-trajectory plotting helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import trimesh  # noqa: E402

from gummybear_validation.milestone_08 import (
    particle_dict_from_catalog_row,
    plot_localization_trajectory_in_mesh,
    resolve_job_stl_path,
)


def test_particle_dict_from_catalog_row():
    row = SimpleNamespace(
        particle_x=1.0,
        particle_y=-2.0,
        particle_z=3.5,
        particle_radius=4.0,
    )
    assert particle_dict_from_catalog_row(row) == {
        "center": (1.0, -2.0, 3.5),
        "radius": 4.0,
    }


def test_resolve_job_stl_path(tmp_path):
    stl = tmp_path / "cad" / "demo.stl"
    stl.parent.mkdir(parents=True)
    stl.write_text("solid demo\nendsolid demo\n")
    job = SimpleNamespace(sequence_id="s", stl_path="cad/demo.stl")
    assert resolve_job_stl_path(job, stl_root=tmp_path) == str(stl)


def test_plot_localization_trajectory_in_mesh_smoke():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=10.0)
    trajectory = np.linspace([-5, 0, 0], [5, 1, 2], 8)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_localization_trajectory_in_mesh(
        ax,
        mesh,
        {"center": [0.0, 0.0, 0.0], "radius": 2.0},
        trajectory,
        target=[0.0, 0.0, 0.0],
        title="smoke",
    )
    plt.close(fig)
