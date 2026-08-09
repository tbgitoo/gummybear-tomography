"""Shared training loops and fusion learning-rate helpers.

Re-exports catalog batching, end-to-end (e2e) fusion training, learning rate
(LR) study orchestration, and illumination-only dataset builders from
:mod:`tomography_ml.training.training_helpers`.

For single-view full-split training, use
:func:`tomography_ml.localization.architecture_capability.train_full_split`.
"""

from .training_helpers import (
    STAGE_B_ANGLE_LR_KEYS,
    STAGE_B_LR_KEYS,
    BatchXyFn,
    batch_from_indices,
    collect_prediction_errors,
    eval_stack,
    load_recommended_stage_b_lrs,
    lr_close,
    make_batch_xy_multiview,
    make_batch_xy_single,
    make_sv_illumination_dataset,
    persist_stage_b_lr_artifacts,
    resolve_run_lr_study,
    run_stage_b_lr_study,
    stage_b_lr_key_is_known,
    train_e2e,
)

__all__ = [
    "STAGE_B_ANGLE_LR_KEYS",
    "STAGE_B_LR_KEYS",
    "BatchXyFn",
    "batch_from_indices",
    "collect_prediction_errors",
    "eval_stack",
    "load_recommended_stage_b_lrs",
    "lr_close",
    "make_batch_xy_multiview",
    "make_batch_xy_single",
    "make_sv_illumination_dataset",
    "persist_stage_b_lr_artifacts",
    "resolve_run_lr_study",
    "run_stage_b_lr_study",
    "stage_b_lr_key_is_known",
    "train_e2e",
]
