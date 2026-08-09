"""Tests for offline workbook particle-centre randomization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gummybear.datasets import (
    DEFAULT_SPLIT_FRACTIONS,
    ParticleRandomizationResult,
    SplitRandomizationResult,
    assign_particle_splits,
    randomize_workbook_particle_centers,
    randomize_workbook_sequence_splits,
    write_example_generation_workbook,
)
from gummybear.datasets.generation_workbook import WorkbookValidationError
from gummybear.geometry import load_stl
from gummybear.particles import sample_random_centers_in_mesh

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_randomize_workbook_particle_centers_reproducible(tmp_path: Path):
    workbook = tmp_path / "plan.xlsx"
    write_example_generation_workbook(workbook)

    first = randomize_workbook_particle_centers(workbook, seed=17, root=REPO_ROOT)
    assert isinstance(first, ParticleRandomizationResult)
    assert first.n_particles == 1
    assert first.seed == 17
    assert first.stl_path == "cad/proto_bear.stl"
    assert first.centers.shape == (1, 3)

    frames_a = pd.read_excel(workbook, sheet_name="particles", engine="openpyxl")
    assert int(frames_a.loc[0, "seed"]) == 17
    center_a = np.array(
        [
            frames_a.loc[0, "center_x"],
            frames_a.loc[0, "center_y"],
            frames_a.loc[0, "center_z"],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(center_a, first.centers[0])

    second = randomize_workbook_particle_centers(workbook, seed=17, root=REPO_ROOT)
    np.testing.assert_allclose(first.centers, second.centers)

    mesh = load_stl(REPO_ROOT / "cad" / "proto_bear.stl")
    assert bool(mesh.contains(first.centers)[0])

    expected = sample_random_centers_in_mesh(mesh, 1, seed=17)
    np.testing.assert_allclose(first.centers, expected)


def test_randomize_multiple_particles_sheet_order(tmp_path: Path):
    workbook = tmp_path / "multi.xlsx"
    write_example_generation_workbook(workbook)
    frames = pd.read_excel(workbook, sheet_name=None, engine="openpyxl")
    particles = frames["particles"]
    extra = particles.iloc[[0]].copy()
    extra.loc[:, "particle_setup_id"] = "particle_smoke_sphere_002"
    extra.loc[:, "center_x"] = 0.0
    extra.loc[:, "center_y"] = 0.0
    extra.loc[:, "center_z"] = 0.0
    frames["particles"] = pd.concat([particles, extra], ignore_index=True)
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    result = randomize_workbook_particle_centers(workbook, seed=9, root=REPO_ROOT)
    assert result.n_particles == 2
    mesh = load_stl(REPO_ROOT / "cad" / "proto_bear.stl")
    assert np.all(mesh.contains(result.centers))
    expected = sample_random_centers_in_mesh(mesh, 2, seed=9)
    np.testing.assert_allclose(result.centers, expected)

    written = pd.read_excel(workbook, sheet_name="particles", engine="openpyxl")
    assert list(written["seed"].astype(int)) == [9, 9]
    for i in range(2):
        np.testing.assert_allclose(
            [
                written.loc[i, "center_x"],
                written.loc[i, "center_y"],
                written.loc[i, "center_z"],
            ],
            expected[i],
        )


def test_only_enabled_skips_disabled_rows(tmp_path: Path):
    workbook = tmp_path / "enabled.xlsx"
    write_example_generation_workbook(workbook)
    frames = pd.read_excel(workbook, sheet_name=None, engine="openpyxl")
    particles = frames["particles"]
    disabled = particles.iloc[[0]].copy()
    disabled.loc[:, "particle_setup_id"] = "particle_disabled"
    disabled.loc[:, "enabled"] = False
    disabled.loc[:, "center_x"] = 1.0
    disabled.loc[:, "center_y"] = 2.0
    disabled.loc[:, "center_z"] = 3.0
    disabled.loc[:, "seed"] = 0
    frames["particles"] = pd.concat([particles, disabled], ignore_index=True)
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    result = randomize_workbook_particle_centers(
        workbook,
        seed=5,
        root=REPO_ROOT,
        only_enabled=True,
    )
    assert result.n_particles == 1
    written = pd.read_excel(workbook, sheet_name="particles", engine="openpyxl")
    assert int(written.loc[1, "seed"]) == 0
    np.testing.assert_allclose(
        [
            written.loc[1, "center_x"],
            written.loc[1, "center_y"],
            written.loc[1, "center_z"],
        ],
        [1.0, 2.0, 3.0],
    )


def test_conflicting_stl_paths_raise(tmp_path: Path):
    workbook = tmp_path / "conflict.xlsx"
    write_example_generation_workbook(workbook)
    frames = pd.read_excel(workbook, sheet_name=None, engine="openpyxl")
    sequences = frames["sequences"]
    extra = sequences.iloc[[0]].copy()
    extra.loc[:, "sequence_id"] = "bear_other"
    extra.loc[:, "stl_path"] = "cad/does_not_matter.stl"
    frames["sequences"] = pd.concat([sequences, extra], ignore_index=True)
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    with pytest.raises(WorkbookValidationError, match="single unique"):
        randomize_workbook_particle_centers(workbook, seed=1, root=REPO_ROOT)


def test_missing_mesh_raises(tmp_path: Path):
    workbook = tmp_path / "missing_mesh.xlsx"
    write_example_generation_workbook(workbook)
    with pytest.raises(WorkbookValidationError, match="Mesh STL not found"):
        randomize_workbook_particle_centers(workbook, seed=1, root=tmp_path)


def _expand_sequences_for_split_test(workbook: Path, *, n_particles: int = 10) -> None:
    """Build n_particles unique particles with 2 sequences each (shared optical variants)."""
    frames = pd.read_excel(workbook, sheet_name=None, engine="openpyxl")
    base_seq = frames["sequences"].iloc[[0]].copy()
    base_part = frames["particles"].iloc[[0]].copy()
    seq_rows = []
    part_rows = []
    for i in range(n_particles):
        pid = f"particle_{i:03d}"
        part = base_part.copy()
        part.loc[:, "particle_setup_id"] = pid
        part_rows.append(part)
        for j in range(2):
            row = base_seq.copy()
            row.loc[:, "sequence_id"] = f"seq_{i:03d}_{j}"
            row.loc[:, "particle_setup_id"] = pid
            row.loc[:, "split"] = "train"
            row.loc[:, "seed"] = 0
            seq_rows.append(row)
    frames["sequences"] = pd.concat(seq_rows, ignore_index=True)
    frames["particles"] = pd.concat(part_rows, ignore_index=True)
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)


def test_assign_particle_splits_default_fractions_70_15_15():
    ids = [f"p{i:03d}" for i in range(300)]
    assignment = assign_particle_splits(ids, seed=42)
    counts = {name: 0 for name in ("train", "validation", "test")}
    for split in assignment.values():
        counts[split] += 1
    assert counts == {"train": 210, "validation": 45, "test": 45}
    assert set(assignment) == set(ids)


def test_randomize_workbook_sequence_splits_stratified_and_reproducible(tmp_path: Path):
    workbook = tmp_path / "splits.xlsx"
    write_example_generation_workbook(workbook)
    _expand_sequences_for_split_test(workbook, n_particles=10)

    first = randomize_workbook_sequence_splits(workbook, seed=42)
    assert isinstance(first, SplitRandomizationResult)
    assert first.seed == 42
    assert first.n_particles == 10
    assert first.n_sequences == 20
    assert first.split_fractions == DEFAULT_SPLIT_FRACTIONS

    written = pd.read_excel(workbook, sheet_name="sequences", engine="openpyxl")
    assert set(written["seed"].astype(int)) == {42}
    # stratification: each particle maps to one split
    by_particle = written.groupby("particle_setup_id")["split"].nunique()
    assert int(by_particle.max()) == 1
    # all sequences for a particle share the assignment
    for pid, split_name in first.particle_splits.items():
        rows = written.loc[written["particle_setup_id"] == pid, "split"]
        assert set(rows.astype(str)) == {split_name}

    second = randomize_workbook_sequence_splits(workbook, seed=42)
    assert second.particle_splits == first.particle_splits
    assert second.split_counts == first.split_counts
    written2 = pd.read_excel(workbook, sheet_name="sequences", engine="openpyxl")
    pd.testing.assert_series_equal(
        written["split"].astype(str),
        written2["split"].astype(str),
        check_names=False,
    )


def test_randomize_workbook_sequence_splits_custom_fractions(tmp_path: Path):
    workbook = tmp_path / "custom.xlsx"
    write_example_generation_workbook(workbook)
    _expand_sequences_for_split_test(workbook, n_particles=10)
    result = randomize_workbook_sequence_splits(
        workbook,
        seed=7,
        train_fraction=0.5,
        validation_fraction=0.3,
        test_fraction=0.2,
    )
    particle_counts = {name: 0 for name in ("train", "validation", "test")}
    for split in result.particle_splits.values():
        particle_counts[split] += 1
    assert particle_counts == {"train": 5, "validation": 3, "test": 2}

