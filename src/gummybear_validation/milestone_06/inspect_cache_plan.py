"""Optional reviewer-facing helpers for dry-run cache execution plans.

Phase 1 only: summarize execution plans for notebook display. No physics.

Notebook / protocol: M6
"""

from __future__ import annotations

from typing import Any

from gummybear.datasets.generation_plan import DryRunSummary, ExecutionPlan
from gummybear.paths import repo_relative_path


def inspect_cache_plan(execution_plan: ExecutionPlan) -> dict[str, Any]:
    """Summarize a dry-run execution plan for cache grouping review.

    Nested structure mirrors optical → particle → diffusion grouping with job
    and frame counts. Paths are portable via :func:`repo_relative_path`.

    Args:
        execution_plan: Dry-run plan from the generation workbook loader.

    Returns:
        dict: Keys ``workbook_path``, ``workbook_sha256``, ``cache_root``,
        ``plans_diffusion_operator_cache``, ``clean_groups``, ``warnings``.
    """
    clean_groups = []
    for clean_group in execution_plan.clean_groups:
        particle_groups = []
        for particle_group in clean_group.particle_groups:
            diffusion_groups = []
            for diffusion_group in particle_group.diffusion_groups:
                diffusion_groups.append(
                    {
                        "diffusion_setup_id": diffusion_group.diffusion_setup_id,
                        "job_count": len(diffusion_group.jobs),
                        "frame_count": len(diffusion_group.camera_tasks),
                        "provenance": diffusion_group.provenance,
                    }
                )
            particle_groups.append(
                {
                    "particle_setup_id": particle_group.particle_setup_id,
                    "particle_group_id": particle_group.particle_group_id,
                    "particle_count": particle_group.particle_count,
                    "particle_source_cache_id": (
                        particle_group.particle_source_cache_id
                    ),
                    "cache_status": particle_group.cache_status,
                    "diffusion_groups": diffusion_groups,
                }
            )
        clean_groups.append(
            {
                "optical_setup_id": clean_group.optical_setup_id,
                "clean_optical_cache_id": clean_group.clean_optical_cache_id,
                "cache_status": clean_group.cache_status,
                "particle_groups": particle_groups,
            }
        )

    return {
        "workbook_path": repo_relative_path(execution_plan.workbook_path),
        "workbook_sha256": execution_plan.workbook_sha256,
        "cache_root": repo_relative_path(execution_plan.cache_root),
        "plans_diffusion_operator_cache": execution_plan.plans_operator_cache,
        "clean_groups": clean_groups,
        "warnings": list(execution_plan.warnings),
    }


def dry_run_summary_table(summary: DryRunSummary) -> dict[str, Any]:
    """Convert a dry-run summary dataclass to a flat dict for pandas display.

    Args:
        summary: Aggregate counts from a generation dry run.

    Returns:
        dict: Same content as ``DryRunSummary.to_dict()``.
    """
    return summary.to_dict()
