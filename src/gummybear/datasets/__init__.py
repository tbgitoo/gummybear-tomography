"""Synthetic multi-view sequence planning, caching, and generation.

Workbooks are a human control surface; typed jobs, cache keys, and manifests
are runtime authority. Train/val/test splits are keyed by ``sequence_id`` only.
"""

from gummybear.datasets.cache_keys import (
    clean_optical_cache_key,
    camera_visibility_cache_key,
    diffusion_settings_provenance,
    particle_source_cache_key,
    phi_sampling_localization_cache_key,
)
from gummybear.datasets.generation_plan import (
    DryRunSummary,
    ExecutionPlan,
    GenerationPlan,
    SequenceJob,
    build_execution_plan,
    default_parallel_workers,
    load_and_summarize_generation_workbook,
    resolve_generation_workers,
    resolve_job_output_identity,
    resolved_job_identity_payload,
    run_generation_plan,
    run_generation_workbook,
    summarize_execution_plan,
    validate_generation_plan,
)
from gummybear.datasets.generation_workbook import (
    M6Workbook,
    WorkbookValidationError,
    attach_particle_group,
    load_generation_workbook,
    write_example_generation_workbook,
    write_generation_workbook_frames,
    write_matrix_generation_workbook,
    write_multi_particle_generation_workbook,
)
from gummybear.datasets.randomization import (
    DEFAULT_SPLIT_FRACTIONS,
    ParticleRandomizationResult,
    SplitRandomizationResult,
    assign_particle_splits,
    randomize_workbook_particle_centers,
    randomize_workbook_sequence_splits,
)
from gummybear.datasets.output_plan import (
    OutputDeltaItem,
    OutputDeltaPlan,
    OutputPlanError,
    build_output_delta_plan,
    reconcile_sequence_output,
    remove_blocking_sequence_outputs,
)
from gummybear.datasets.source_cache import CacheEvent, SourceCacheStore

__all__ = [
    "CacheEvent",
    "DEFAULT_SPLIT_FRACTIONS",
    "DryRunSummary",
    "ExecutionPlan",
    "GenerationPlan",
    "M6Workbook",
    "OutputDeltaItem",
    "OutputDeltaPlan",
    "OutputPlanError",
    "ParticleRandomizationResult",
    "SequenceJob",
    "SourceCacheStore",
    "SplitRandomizationResult",
    "WorkbookValidationError",
    "assign_particle_splits",
    "attach_particle_group",
    "build_execution_plan",
    "build_output_delta_plan",
    "camera_visibility_cache_key",
    "clean_optical_cache_key",
    "default_parallel_workers",
    "diffusion_settings_provenance",
    "load_and_summarize_generation_workbook",
    "load_generation_workbook",
    "particle_source_cache_key",
    "phi_sampling_localization_cache_key",
    "randomize_workbook_particle_centers",
    "randomize_workbook_sequence_splits",
    "reconcile_sequence_output",
    "remove_blocking_sequence_outputs",
    "resolve_generation_workers",
    "resolve_job_output_identity",
    "resolved_job_identity_payload",
    "run_generation_plan",
    "run_generation_workbook",
    "summarize_execution_plan",
    "validate_generation_plan",
    "write_example_generation_workbook",
    "write_generation_workbook_frames",
    "write_matrix_generation_workbook",
    "write_multi_particle_generation_workbook",
]
