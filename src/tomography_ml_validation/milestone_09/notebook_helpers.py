"""Notebook-facing helpers for Milestone 9 camera-view fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tomography_ml.studies.m9_expert_xyz_mean import (
    M9ExpertXyzMeanResult,
    _make_split_dataset,
)
from tomography_ml.studies.study_checkpoints import (
    study_checkpoint_policy,
    study_results_dir,
)
from tomography_ml_validation.milestone_08.notebook_helpers import (
    describe_paths,
    m8_corpus_paths,
)

FINAL_REPORT_REL = "GummyBearTomography_Final_Report.ipynb"
M9_PLAN_REL = "plans/milestone_09/09_camera_view_fusion_plan.md"


def m9_corpus_paths(
    repo_root: Path | str,
    *,
    data_mode: str = "full",
    read_checkpoints: bool = True,
) -> dict[str, Any]:
    """Resolve M8 corpus paths plus M9 checkpoint / schedule defaults."""
    root = Path(repo_root)
    paths = m8_corpus_paths(root, data_mode=data_mode)
    policy = study_checkpoint_policy(
        repo_root=root,
        milestone="m9",
        data_mode=data_mode,
        read_checkpoints=read_checkpoints,
    )
    results_dir = study_results_dir(policy, fallback=paths["output_root"])
    mode = str(data_mode).strip().lower()
    demo = mode in {"inspect", "demo"}
    return {
        **paths,
        "results_dir": results_dir,
        "policy": policy,
        "num_epochs": 40 if demo else 200,
        "early_stop_patience": 25 if demo else 40,
        "angle_stride_deg": 90.0 if demo else 60.0,
        "angle_stride_deg_09_0": 90.0 if demo else 10.0,
        "run_lr_study": True,
        "load_existing": bool(policy.load_existing),
        "retrain": bool(policy.retrain),
    }


def m9_0_split_dataset(result: M9ExpertXyzMeanResult, *, split: str = "validation"):
    """Rebuild a catalog task dataset for 09_0 diagnostic plots."""
    extra = result.extra
    return _make_split_dataset(
        extra["catalog_rows"],
        split=split,
        keep_angles_deg=list(result.view_angles),
        x_field=str(extra["x_field"]),
        image_normalize=str(extra["block"].image_normalize),
        optical_setup_id=str(extra["block"].optical_setup_id_reference),
    )
