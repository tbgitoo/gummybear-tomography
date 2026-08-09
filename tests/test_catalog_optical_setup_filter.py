"""CatalogRow exposes optical_setup_id for DatasetTaskSpec row_filter."""

from __future__ import annotations

from pathlib import Path

from tomography_ml.gummybear_data_catalog import CatalogRow, build_task_dataset
from tomography_ml.gummybear_data_catalog.task_dataset import DatasetTaskSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _row(*, sample_id: int, optical_setup_id: str, split: str = "train") -> CatalogRow:
    return CatalogRow(
        sample_id=sample_id,
        sequence_id=f"seq_{sample_id}",
        split=split,
        output_root=str(REPO_ROOT / "data" / "generated" / "m8_1"),
        sequence_dir=str(
            REPO_ROOT / "data" / "generated" / "m8_1" / "single_particle" / f"seq_{sample_id}"
        ),
        manifest_path=str(
            REPO_ROOT
            / "data"
            / "generated"
            / "m8_1"
            / "single_particle"
            / f"seq_{sample_id}"
            / "manifest.json"
        ),
        field_status="complete",
        schema_version="1.4-m6-draft",
        resolved_job_hash="a" * 64,
        camera_schedule_id="orbit_36",
        frame_count=1,
        angles_deg=(0.0,),
        angles_hash="b" * 64,
        observed_ref=None,
        clean_ref=None,
        particle_ref=None,
        anomaly_ref=None,
        optical_setup_id=optical_setup_id,
        bear_mu_s=0.1,
        bear_mu_a=0.01,
        particle_present=True,
        n_particles=1,
        particle_group_id="",
        particles=(),
        particle_x=1.0,
        particle_y=2.0,
        particle_z=3.0,
        particle_radius=3.0,
        particle_mu_s=10.0,
        particle_mu_a=0.0,
        diffusion_setup_id="diff",
        extrapolation_length=1.0,
        image_domain="camera_intensity",
        composition_domain=None,
    )


def test_row_filter_by_optical_setup_id() -> None:
    rows = [
        _row(sample_id=0, optical_setup_id="opt_m8_high_001"),
        _row(sample_id=1, optical_setup_id="opt_m8_low_001"),
        _row(sample_id=2, optical_setup_id="opt_m8_high_001"),
    ]
    dataset = build_task_dataset(
        rows,
        DatasetTaskSpec(
            name="localization",
            row_filter={
                "split": "train",
                "field_status": "complete",
                "optical_setup_id": "opt_m8_high_001",
            },
            x_fields=("particle_ref",),
            y_fields=("particle_x", "particle_y", "particle_z"),
        ),
    )
    assert len(dataset) == 2
    assert {row.optical_setup_id for row in dataset.rows} == {"opt_m8_high_001"}
