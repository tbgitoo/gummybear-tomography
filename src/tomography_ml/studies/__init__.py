"""Reusable single-view localisation study protocols (WIN 3A-style).

Notebooks should call these helpers rather than inlining Adam/MSE loops.
"""

from .single_view_m8 import (
    ARCH_COLORS,
    ARCH_ORDER,
    CANONICAL_LR_BY_ARCH,
    DEFAULT_LR_GRID,
    DEFAULT_SENSITIVITY_SPLIT_SEEDS,
    LearningRateStudyResult,
    SplitSensitivityStudyResult,
    TrainValTestStudyResult,
    dummy_batch_from_dataset,
    make_win3a_model,
    particle_setup_id_for_row,
    probe_win3a_parameter_counts,
    relabel_catalog_rows_for_split_seed,
    rmse_metrics_from_l2_errors,
    run_learning_rate_study,
    run_split_sensitivity_study,
    run_train_val_test_study,
    select_lr_by_arch,
    set_train_seed,
)

__all__ = [
    "ARCH_COLORS",
    "ARCH_ORDER",
    "CANONICAL_LR_BY_ARCH",
    "DEFAULT_LR_GRID",
    "DEFAULT_SENSITIVITY_SPLIT_SEEDS",
    "LearningRateStudyResult",
    "SplitSensitivityStudyResult",
    "TrainValTestStudyResult",
    "dummy_batch_from_dataset",
    "make_win3a_model",
    "particle_setup_id_for_row",
    "probe_win3a_parameter_counts",
    "relabel_catalog_rows_for_split_seed",
    "rmse_metrics_from_l2_errors",
    "run_learning_rate_study",
    "run_split_sensitivity_study",
    "run_train_val_test_study",
    "select_lr_by_arch",
    "set_train_seed",
]
