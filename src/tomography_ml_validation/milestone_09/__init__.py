"""Installed pytest contracts and notebook helpers for Milestone 9."""

from tomography_ml.studies.m9_expert_xyz_mean import (
    M09_EXPERT_XYZ_MEAN,
    M9ExpertXyzMeanConfig,
    M9ExpertXyzMeanResult,
    assert_affine_identity_shared_linear,
    run_m9_expert_xyz_mean,
)
from tomography_ml.studies.m9_e2e_geometry_fusion import (
    M9E2EConfig,
    run_m9_e2e_geometry_fusion_family,
)
from tomography_ml.studies.m9_frozen_fusion import (
    M9FusionConfig,
    run_m9_frozen_fusion_family,
)
from tomography_ml_validation.milestone_09.notebook_helpers import (
    FINAL_REPORT_REL,
    M9_PLAN_REL,
    describe_paths,
    m9_0_split_dataset,
    m9_corpus_paths,
)
from tomography_ml_validation.milestone_09.validation import (
    test_m9_0_affine_identity_shared_linear,
    test_m9_0_fusion_pattern_is_expert_mean,
    test_m9_1_frozen_fusion_pattern_ids,
    test_m9_2_compact_vs_09_3_capacity,
    test_m9_corpus_paths_resolve,
)
from tomography_ml_validation.plotting.m9_expert_xyz_mean import (
    collect_m9_0_bias_vs_std,
    plot_m9_0_bias_vs_expert_std,
    plot_m9_0_per_angle_experts,
)

__all__ = [
    "FINAL_REPORT_REL",
    "M09_EXPERT_XYZ_MEAN",
    "M9E2EConfig",
    "M9ExpertXyzMeanConfig",
    "M9ExpertXyzMeanResult",
    "M9FusionConfig",
    "M9_PLAN_REL",
    "assert_affine_identity_shared_linear",
    "collect_m9_0_bias_vs_std",
    "describe_paths",
    "m9_0_split_dataset",
    "m9_corpus_paths",
    "plot_m9_0_bias_vs_expert_std",
    "plot_m9_0_per_angle_experts",
    "run_m9_e2e_geometry_fusion_family",
    "run_m9_expert_xyz_mean",
    "run_m9_frozen_fusion_family",
    "test_m9_0_affine_identity_shared_linear",
    "test_m9_0_fusion_pattern_is_expert_mean",
    "test_m9_1_frozen_fusion_pattern_ids",
    "test_m9_2_compact_vs_09_3_capacity",
    "test_m9_corpus_paths_resolve",
]
