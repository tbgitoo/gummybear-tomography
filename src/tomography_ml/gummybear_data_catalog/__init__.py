"""Data catalog: flat rows, lazy role loading, and task datasets.

Joins workbook ``SequenceJob`` definitions with optional on-disk manifests to
build :class:`CatalogRow` samples. Task datasets resolve ``RoleRef`` fields
lazily; default image loading uses float32 ``.raw.tif`` sidecars.
"""


from .gummybear_adapter import (
    ALLOWED_FIELD_STATUSES,
    ALLOWED_SCHEDULE_STATUSES,
    REQUIRED_SCHEDULE_IDENTITY_COLUMNS,
    angles_hash,
    catalog_jobs_to_dataframe,
    filter_schedule_consistent,
    load_catalog_jobs,
    load_catalog_plan,
    ordered_angles_deg,
    reconcile_catalog_jobs_with_manifest,
    schedule_identity_table,
)

from .catalog import ( 
    RoleRef,  
    CatalogRow, 
    ParticleLabel,
    build_catalog_rows 
)

from .task_dataset import (
    DEFAULT_IMAGE_NORMALIZE,
    DEFAULT_IMAGE_REPRESENTATION,
    IMAGE_NORMALIZE_NONE,
    IMAGE_NORMALIZE_PER_IMAGE_MINMAX,
    IMAGE_NORMALIZE_PER_IMAGE_ZSCORE,
    IMAGE_NORMALIZE_TRAIN_SPLIT_ZSCORE,
    IMAGE_REPRESENTATION_JPEG_UINT8,
    IMAGE_REPRESENTATION_RAW_FLOAT,
    M7_IMAGE_REPRESENTATION,
    CatalogTaskDataset,
    DatasetTaskSpec,
    IntensityStats,
    apply_image_normalize,
    build_task_dataset,
    estimate_intensity_stats,
    load_role_array,
    resolve_task_sample,
)

from .IlluminationOnlyDataset import (
    IlluminationOnlyDataset,
    build_illumination_joint_groups,
    count_groups_by_split,
    groups_for_split,
    particle_id_from_sequence_id,
)
from .HierarchicalCameraLightDataset import HierarchicalCameraLightDataset


__all__ = [
    "ALLOWED_FIELD_STATUSES",
    "ALLOWED_SCHEDULE_STATUSES",
    "REQUIRED_SCHEDULE_IDENTITY_COLUMNS",
    "angles_hash",
    "catalog_jobs_to_dataframe",
    "filter_schedule_consistent",
    "load_catalog_jobs",
    "load_catalog_plan",
    "ordered_angles_deg",
    "reconcile_catalog_jobs_with_manifest",
    "schedule_identity_table",
    "RoleRef",
    "CatalogRow",
    "ParticleLabel",
    "build_catalog_rows",
    "CatalogTaskDataset",
    "DatasetTaskSpec",
    "DEFAULT_IMAGE_NORMALIZE",
    "DEFAULT_IMAGE_REPRESENTATION",
    "IMAGE_NORMALIZE_NONE",
    "IMAGE_NORMALIZE_PER_IMAGE_MINMAX",
    "IMAGE_NORMALIZE_PER_IMAGE_ZSCORE",
    "IMAGE_NORMALIZE_TRAIN_SPLIT_ZSCORE",
    "IMAGE_REPRESENTATION_JPEG_UINT8",
    "IMAGE_REPRESENTATION_RAW_FLOAT",
    "IntensityStats",
    "M7_IMAGE_REPRESENTATION",
    "apply_image_normalize",
    "build_task_dataset",
    "estimate_intensity_stats",
    "load_role_array",
    "resolve_task_sample",
    "IlluminationOnlyDataset",
    "HierarchicalCameraLightDataset",
    "build_illumination_joint_groups",
    "count_groups_by_split",
    "groups_for_split",
    "particle_id_from_sequence_id",
]

