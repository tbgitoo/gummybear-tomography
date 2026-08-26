"""Notebook-facing helpers for Milestone 10 illumination / hierarchical fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tomography_ml.studies.study_checkpoints import (
    study_checkpoint_policy,
    study_results_dir,
)
from tomography_ml_validation.milestone_08.notebook_helpers import describe_paths

FINAL_REPORT_REL = "GummyBearTomography_Final_Report.ipynb"
M10_PLAN_REL = "plans/milestone_10/10_lighting_fusion_plan.md"


def m10_corpus_paths(
    repo_root: Path | str,
    *,
    data_mode: str = "full",
    read_checkpoints: bool = True,
) -> dict[str, Any]:
    """Resolve M10 workbook / corpus paths plus checkpoint / schedule defaults."""
    root = Path(repo_root)
    mode = str(data_mode).strip().lower()
    demo = mode in {"inspect", "demo"}
    if demo:
        workbook = root / "configs/m10/m10_demo.xlsx"
        output_root = root / "data/generated/m10_demo"
    else:
        workbook = root / "configs/m10/localization_m10_illumination.xlsx"
        output_root = root / "data/generated/m10_illumination"
    policy = study_checkpoint_policy(
        repo_root=root,
        milestone="m10",
        data_mode=data_mode,
        read_checkpoints=read_checkpoints,
    )
    results_dir = study_results_dir(policy, fallback=output_root)
    return {
        "repo_root": root,
        "workbook_path": workbook,
        "output_root": output_root,
        "cache_root": output_root / "_cache",
        "results_dir": results_dir,
        "policy": policy,
        "num_epochs": 40 if demo else 200,
        "early_stop_patience": 25 if demo else 40,
        "angle_stride_deg": 90.0 if demo else 10.0,
        "batch_size": 2 if demo else 1,
        "run_lr_study": bool(demo),
        "load_existing": bool(policy.load_existing),
        "retrain": bool(policy.retrain),
    }


__all__ = [
    "FINAL_REPORT_REL",
    "M10_PLAN_REL",
    "describe_paths",
    "m10_corpus_paths",
]
