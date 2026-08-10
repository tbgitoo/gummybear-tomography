"""Unit tests for WIN 3A single-view study helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tomography_ml.gummybear_data_catalog.task_dataset import DatasetTaskSpec
from tomography_ml.studies import (
    ARCH_ORDER,
    CANONICAL_LR_BY_ARCH,
    M8_CANONICAL_LR_BY_ARCH,
    make_m8_single_view_model,
    probe_m8_parameter_counts,
    relabel_catalog_rows_for_split_seed,
    rmse_metrics_from_l2_errors,
    run_learning_rate_study,
    run_split_sensitivity_study,
    run_train_val_test_study,
    select_lr_by_arch,
)


class _TinyCatalogDS(torch.utils.data.Dataset):
    """Minimal catalog-shaped dataset used only inside patched build_task_dataset."""

    def __init__(self, n: int, *, z_only: bool):
        self.n = int(n)
        self.z_only = bool(z_only)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        # Shape matches single-view keep_angles: (V=1, C=1, H, W)
        x = {"anomaly_ref": np.zeros((1, 1, 8, 8), dtype=np.float32) + float(index)}
        if self.z_only:
            y = {"particle_z": float(index)}
        else:
            y = {
                "particle_x": float(index),
                "particle_y": float(index) + 0.5,
                "particle_z": float(index) + 1.0,
            }
        return x, y


def test_make_m8_single_view_model_and_rmse_metrics() -> None:
    device = torch.device("cpu")
    for arch in ARCH_ORDER:
        model = make_m8_single_view_model(arch, n_outputs=1, device=device)
        out = model(torch.zeros(2, 1, 8, 8))
        assert out.shape == (2, 1)

    z_metrics = rmse_metrics_from_l2_errors(
        np.asarray([3.0, 4.0]), y_fields=("particle_z",)
    )
    assert np.isclose(z_metrics["train_RMSE_total"], float(np.sqrt(np.mean([9.0, 16.0]))))
    assert np.isclose(z_metrics["train_RMSE_Z"], z_metrics["train_RMSE_total"])
    assert np.isnan(z_metrics["train_RMSE_X"])

    xyz_metrics = rmse_metrics_from_l2_errors(
        np.asarray([1.0, 1.0]), y_fields=("particle_x", "particle_y", "particle_z")
    )
    assert np.isclose(xyz_metrics["train_RMSE_total"], 1.0)
    assert np.isnan(xyz_metrics["train_RMSE_X"])


def test_select_lr_by_arch() -> None:
    lr_results = {
        "pooled": {
            1e-3: {"val_mse": 2.0},
            1e-2: {"val_mse": 1.0},
            1e-1: {"val_mse": float("nan")},
        },
        "fourier": {1e-3: {"val_mse": 0.5}, 1e-2: {"val_mse": 0.8}},
        "flatten": {1e-3: {"val_mse": 0.9}, 3e-4: {"val_mse": 0.4}},
    }
    selected = select_lr_by_arch(lr_results)
    assert selected["pooled"] == 1e-2
    assert selected["fourier"] == 1e-3
    assert selected["flatten"] == 3e-4
    assert CANONICAL_LR_BY_ARCH["fourier"] == 0.03
    assert M8_CANONICAL_LR_BY_ARCH["fourier"] == 0.03


def test_run_lr_and_train_val_helpers(tmp_path: Path, monkeypatch) -> None:
    import tomography_ml.studies.single_view_m8 as study_mod

    z_task = DatasetTaskSpec(
        name="localization_z",
        row_filter={"split": "train"},
        x_fields=("anomaly_ref",),
        y_fields=("particle_z",),
    )

    def _fake_build(catalog_rows, task):
        split = str(task.row_filter.get("split", "train"))
        n = {"train": 6, "validation": 4, "test": 4}[split]
        return _TinyCatalogDS(n, z_only=True)

    monkeypatch.setattr(study_mod, "build_task_dataset", _fake_build)

    device = torch.device("cpu")
    lr_result = run_learning_rate_study(
        catalog_rows=[{}],
        task=z_task,
        device=device,
        lr_grid=(1e-3, 1e-2),
        num_epochs=2,
        batch_size=4,
        seed=0,
        results_dir=tmp_path / "lr_study",
        load_existing=False,
    )
    assert set(lr_result.lr_by_arch) == set(ARCH_ORDER)
    assert lr_result.train_size == 6
    assert lr_result.val_size == 4
    assert lr_result.checkpoint_path is not None
    assert lr_result.checkpoint_path.is_file()
    assert not lr_result.skipped_train
    for arch in ARCH_ORDER:
        for lr in (1e-3, 1e-2):
            curve = lr_result.lr_results[arch][lr]["train_curve"]
            assert len(curve) == 2
            assert lr in lr_result.final_state_by_arch_lr[arch]
            assert "encoder.stem.0.weight" in lr_result.final_state_by_arch_lr[arch][lr] or any(
                k.endswith("weight") for k in lr_result.final_state_by_arch_lr[arch][lr]
            )

    # Selected-LR weights reload into a fresh model.
    best_lr = lr_result.lr_by_arch["fourier"]
    model = make_m8_single_view_model("fourier", n_outputs=1, device=device)
    model.load_state_dict(lr_result.final_state_for("fourier"))
    with torch.no_grad():
        out = model(torch.zeros(2, 1, 8, 8))
    assert out.shape == (2, 1)

    lr_loaded = run_learning_rate_study(
        catalog_rows=[{}],
        task=z_task,
        device=device,
        lr_grid=(1e-3, 1e-2),
        num_epochs=2,
        batch_size=4,
        seed=0,
        results_dir=tmp_path / "lr_study",
        load_existing=True,
        retrain=False,
    )
    assert lr_loaded.skipped_train
    assert lr_loaded.lr_by_arch == lr_result.lr_by_arch
    assert lr_loaded.lr_results["fourier"][1e-3]["val_mse"] == (
        lr_result.lr_results["fourier"][1e-3]["val_mse"]
    )
    loaded_state = lr_loaded.final_state_for("fourier", best_lr)
    for key, tensor in lr_result.final_state_for("fourier", best_lr).items():
        assert torch.equal(tensor, loaded_state[key])

    tv = run_train_val_test_study(
        catalog_rows=[{}],
        task=z_task,
        device=device,
        results_dir=tmp_path / "study",
        lr_by_arch=lr_result.lr_by_arch,
        notebook_id="test",
        experiment_id="test_exp",
        variant_prefix="m_test",
        num_epochs=2,
        batch_size=4,
        n_repeat_training=2,
        base_seed=0,
        checkpoint_name="m08_train_val_test_z.pt",
        load_existing=False,
    )
    assert tv.n_rep == 2
    assert tv.history_path.is_file()
    assert tv.session_summary_path.is_file()
    assert tv.comparison_path.is_file()
    assert set(tv.full_results) == set(ARCH_ORDER)
    assert len(tv.session_summary_df) == 3
    assert tv.checkpoint_path is not None and tv.checkpoint_path.is_file()
    assert set(tv.final_state_by_arch) == set(ARCH_ORDER)
    assert not tv.skipped_train

    tv_loaded = run_train_val_test_study(
        catalog_rows=[{}],
        task=z_task,
        device=device,
        results_dir=tmp_path / "study",
        lr_by_arch=lr_result.lr_by_arch,
        notebook_id="test",
        experiment_id="test_exp",
        variant_prefix="m_test",
        num_epochs=2,
        batch_size=4,
        n_repeat_training=2,
        base_seed=0,
        checkpoint_name="m08_train_val_test_z.pt",
        load_existing=True,
        retrain=False,
    )
    assert tv_loaded.skipped_train
    assert set(tv_loaded.final_state_by_arch) == set(ARCH_ORDER)
    for arch in ARCH_ORDER:
        for key, tensor in tv.final_state_by_arch[arch].items():
            assert torch.equal(tensor, tv_loaded.final_state_by_arch[arch][key])


def _fake_catalog_row(sample_id: int, particle_setup_id: str, split: str):
    from tomography_ml.gummybear_data_catalog.catalog import CatalogRow, ParticleLabel

    label = ParticleLabel(
        particle_setup_id=particle_setup_id,
        center_x=0.0,
        center_y=0.0,
        center_z=float(sample_id),
        radius=1.0,
    )
    return CatalogRow(
        sample_id=sample_id,
        sequence_id=f"seq_{particle_setup_id}",
        split=split,
        output_root="/tmp",
        sequence_dir="/tmp/seq",
        manifest_path="/tmp/seq/manifest.json",
        field_status="complete",
        schema_version="1.6",
        resolved_job_hash="hash",
        camera_schedule_id="cam",
        frame_count=1,
        angles_deg=(180.0,),
        angles_hash="ah",
        observed_ref=None,
        clean_ref=None,
        particle_ref=None,
        anomaly_ref=None,
        optical_setup_id="opt_m8_high_001",
        bear_mu_s=1.0,
        bear_mu_a=0.1,
        particle_present=True,
        n_particles=1,
        particle_group_id="g",
        particles=(label,),
        particle_x=0.0,
        particle_y=0.0,
        particle_z=float(sample_id),
        particle_radius=1.0,
        particle_mu_s=None,
        particle_mu_a=None,
        diffusion_setup_id="diff",
        extrapolation_length=1.0,
        image_domain="camera_intensity",
        composition_domain=None,
    )


def test_relabel_catalog_rows_for_split_seed() -> None:
    rows = [
        _fake_catalog_row(i, f"particle_{i:03d}", "train")
        for i in range(10)
    ]
    relabeled = relabel_catalog_rows_for_split_seed(rows, seed=60)
    assert len(relabeled) == 10
    splits = {r.split for r in relabeled}
    assert splits == {"train", "validation", "test"}
    # Same seed is deterministic.
    again = relabel_catalog_rows_for_split_seed(rows, seed=60)
    assert [r.split for r in again] == [r.split for r in relabeled]
    # Different seed changes membership for this particle count.
    other = relabel_catalog_rows_for_split_seed(rows, seed=61)
    assert [r.split for r in other] != [r.split for r in relabeled]


def test_run_split_sensitivity_study(tmp_path: Path, monkeypatch) -> None:
    import tomography_ml.studies.single_view_m8 as study_mod

    xyz_task = DatasetTaskSpec(
        name="localization_xyz",
        row_filter={"split": "train"},
        x_fields=("anomaly_ref",),
        y_fields=("particle_x", "particle_y", "particle_z"),
    )

    def _fake_build(catalog_rows, task):
        split = str(task.row_filter.get("split", "train"))
        n = {"train": 6, "validation": 4, "test": 4}[split]
        return _TinyCatalogDS(n, z_only=False)

    monkeypatch.setattr(study_mod, "build_task_dataset", _fake_build)

    rows = [_fake_catalog_row(i, f"particle_{i:03d}", "train") for i in range(20)]
    sens = run_split_sensitivity_study(
        catalog_rows=rows,
        task=xyz_task,
        device=torch.device("cpu"),
        results_dir=tmp_path / "sens",
        lr_by_arch=dict(M8_CANONICAL_LR_BY_ARCH),
        split_seeds=(60, 61),
        num_epochs=1,
        batch_size=4,
        training_seed=0,
        verbose=False,
    )
    assert sens.split_seeds == (60, 61)
    assert set(sens.per_seed_studies) == {60, 61}
    assert len(sens.per_seed_metrics) == 2 * 3
    assert len(sens.summary_df) == 3
    assert sens.summary_path.is_file()
    assert sens.per_seed_path.is_file()
    assert (tmp_path / "sens" / "split_seed_60" / "run_history.csv").is_file()
