"""Tests for CatalogTaskDataset readable __repr__."""

from __future__ import annotations

from pathlib import Path

from tomography_ml.gummybear_data_catalog import CatalogRow, build_task_dataset
from tomography_ml.gummybear_data_catalog.task_dataset import DatasetTaskSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _row(
    *,
    sample_id: int,
    sequence_id: str,
    split: str = "train",
    angles_deg: tuple[float, ...] = (0.0, 180.0),
    particle_x: float | None = 1.0,
    particle_y: float | None = 2.0,
    particle_z: float | None = 3.0,
) -> CatalogRow:
    return CatalogRow(
        sample_id=sample_id,
        sequence_id=sequence_id,
        split=split,
        output_root=str(REPO_ROOT / "data" / "generated" / "m8_1"),
        sequence_dir=str(
            REPO_ROOT / "data" / "generated" / "m8_1" / "single_particle" / sequence_id
        ),
        manifest_path=str(
            REPO_ROOT
            / "data"
            / "generated"
            / "m8_1"
            / "single_particle"
            / sequence_id
            / "manifest.json"
        ),
        field_status="complete",
        schema_version="1.4-m6-draft",
        resolved_job_hash="a" * 64,
        camera_schedule_id="orbit_36",
        frame_count=len(angles_deg),
        angles_deg=angles_deg,
        angles_hash="b" * 64,
        observed_ref=None,
        clean_ref=None,
        particle_ref=None,
        anomaly_ref=None,
        optical_setup_id="opt_test",
        bear_mu_s=0.1,
        bear_mu_a=0.01,
        particle_present=particle_x is not None,
        n_particles=1 if particle_x is not None else 0,
        particle_group_id="",
        particles=(),
        particle_x=particle_x,
        particle_y=particle_y,
        particle_z=particle_z,
        particle_radius=3.0 if particle_x is not None else None,
        particle_mu_s=10.0 if particle_x is not None else None,
        particle_mu_a=0.0 if particle_x is not None else None,
        diffusion_setup_id="diff",
        extrapolation_length=1.0,
        image_domain="camera_intensity",
        composition_domain=None,
    )


def test_catalog_task_dataset_repr_is_readable_and_repo_relative():
    rows = [
        _row(sample_id=0, sequence_id="seq_a"),
        _row(sample_id=1, sequence_id="seq_b", split="val"),
    ]
    dataset = build_task_dataset(
        rows,
        DatasetTaskSpec(
            name="localization",
            row_filter={"split": "train", "field_status": "complete"},
            x_fields=("particle_ref",),
            y_fields=("particle_x", "particle_y", "particle_z"),
            keep_angles_deg=180,
        ),
    )
    text = repr(dataset)

    assert "CatalogTaskDataset(" in text
    assert "task='localization'" in text
    assert "n=1" in text
    assert "keep_angles_deg=(180.0,)" in text
    assert "seq_a" in text
    assert "seq_b" not in text
    assert "sequence_id" in text
    assert "data/generated/m8_1/single_particle/seq_a" in text
    assert str(Path.home()) not in text
    assert "/Users/" not in text
    assert "note=views subset to angles [180]" in text


def test_catalog_task_dataset_repr_truncates_long_tables():
    rows = [
        _row(sample_id=i, sequence_id=f"seq_{i:03d}")
        for i in range(25)
    ]
    dataset = build_task_dataset(
        rows,
        DatasetTaskSpec(
            name="localization",
            row_filter={},
            x_fields=("sequence_id",),
            y_fields=("particle_x",),
        ),
    )
    text = repr(dataset)
    assert "n=25" in text
    assert "seq_000" in text
    assert "... (5 more)" in text
