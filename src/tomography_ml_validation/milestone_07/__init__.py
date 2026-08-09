"""Installed pytest contracts for catalog membership and task-dataset extraction."""

from .validation import (
    test_m7_1_catalog_membership_contract,
    test_m7_2_manifest_reconciliation_reports_artifact_readiness_without_changing_catalog_membership,
    test_m7_3_schedule_consistent_subset_rejects_mixed_acquisition_shapes,
    test_m7_4_flat_catalog_joins_jobs_and_manifests_without_loading_tensors,
    test_m7_5_task_dataset_returns_lazy_xy_without_redefining_sample_as_tensor,
    test_m7_6_workbook_catalog_and_subset_support_dual_task_views,
)

__all__ = [
    "test_m7_1_catalog_membership_contract",
    "test_m7_2_manifest_reconciliation_reports_artifact_readiness_without_changing_catalog_membership",
    "test_m7_3_schedule_consistent_subset_rejects_mixed_acquisition_shapes",
    "test_m7_4_flat_catalog_joins_jobs_and_manifests_without_loading_tensors",
    "test_m7_5_task_dataset_returns_lazy_xy_without_redefining_sample_as_tensor",
    "test_m7_6_workbook_catalog_and_subset_support_dual_task_views",
]
