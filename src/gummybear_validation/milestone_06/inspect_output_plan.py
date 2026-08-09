"""Reviewer-facing summaries for output delta plans (requested, disabled, orphaned).

Notebook / protocol: M6.5
"""

from __future__ import annotations

from typing import Any

from gummybear.datasets.output_plan import OutputDeltaPlan
from gummybear.paths import repo_relative_path


def inspect_output_delta_plan(plan: OutputDeltaPlan) -> list[dict[str, Any]]:
    """Flatten an output delta plan into notebook-friendly rows.

    Each row describes one sequence in ``requested``, ``disabled``, or
    ``orphaned`` categories with portable ``output_path`` strings.

    Args:
        plan: Parsed :class:`gummybear.datasets.output_plan.OutputDeltaPlan`.

    Returns:
        list[dict]: Rows with keys ``category``, ``sequence_id``, ``status``,
        ``reason``, ``output_path``, ``details``.
    """
    rows: list[dict[str, Any]] = []
    for category, items in (
        ("requested", plan.requested),
        ("disabled", plan.disabled),
        ("orphaned", plan.orphaned),
    ):
        for item in items:
            rows.append(
                {
                    "category": category,
                    "sequence_id": item.sequence_id,
                    "status": item.status,
                    "reason": item.reason,
                    "output_path": repo_relative_path(item.output_path),
                    "details": dict(item.details),
                }
            )
    return rows
