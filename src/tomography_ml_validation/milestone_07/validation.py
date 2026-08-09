"""Installed pytest contracts for catalog membership and task-dataset extraction."""

from __future__ import annotations

from importlib.resources import as_file, files

import numpy as np
import pytest

from tomography_ml.gummybear_data_catalog import (
    ALLOWED_FIELD_STATUSES,
    ALLOWED_SCHEDULE_STATUSES,
    M7_IMAGE_REPRESENTATION,
    REQUIRED_SCHEDULE_IDENTITY_COLUMNS,
    CatalogRow,
    DatasetTaskSpec,
    RoleRef,
    angles_hash,
    build_catalog_rows,
    build_task_dataset,
    catalog_jobs_to_dataframe,
    filter_schedule_consistent,
    load_catalog_jobs,
    load_catalog_plan,
    ordered_angles_deg,
    reconcile_catalog_jobs_with_manifest,
    schedule_identity_table,
)


# Importlib resource path to the installed validation workbook and fixtures.
VALIDATION_ROOT = files("tomography_ml_validation") / "test_data"

# Columns every manifest reconciliation dataframe must expose.
REQUIRED_RECONCILIATION_COLUMNS = (
    "sample_id",
    "sequence_id",
    "sequence_dir_exists",
    "manifest_exists",
    "manifest_sequence_id",
    "schema_version",
    "frame_count",
    "roles_present",
    "field_status",
)


@pytest.mark.milestone("M7.1")
@pytest.mark.proves(
    "Enabled GenerationPlan.jobs become catalog rows 1:1; "
    "disabled sequence IDs do not appear."
)
def test_m7_1_catalog_membership_contract():
    """
    Enabled workbook jobs become catalog rows one-to-one; disabled IDs stay out.

    Contract:
    - The installed workbook defines the GenerationPlan.
    - Every enabled GenerationPlan job becomes one catalog row.
    - Disabled workbook rows remain reported in plan.disabled_sequence_ids.
    - Disabled rows do not become catalog rows.
    - Catalog row order follows plan.jobs.
    - Catalog membership is not inferred from generated folders or cache folders.

    Notebook / protocol: M7.1
    """

    with as_file(VALIDATION_ROOT) as validation_root:
        workbook_path = (
            validation_root
            / "configs"
            / "m6"
            / "m6_matrix_plan.xlsx"
        )

        output_root = validation_root / "data" / "generated" / "m6_5"
        cache_root = output_root / "_cache"

        assert workbook_path.exists(), (
            "Installed validation workbook is missing: "
            f"{workbook_path}"
        )

        assert output_root.exists(), (
            "Installed generated validation folder is missing: "
            f"{output_root}"
        )

        assert cache_root.exists(), (
            "Installed cache validation folder is missing: "
            f"{cache_root}"
        )

        plan = load_catalog_plan(
            workbook_path=workbook_path,
            root_path=validation_root,
        )
        jobs = load_catalog_jobs(
            workbook_path=workbook_path,
            root_path=validation_root,
        )

        catalog_df = catalog_jobs_to_dataframe(jobs)

    expected_sequence_ids = [
        job.sequence_id
        for job in plan.jobs
    ]

    observed_sequence_ids = catalog_df["sequence_id"].tolist()

    assert observed_sequence_ids == expected_sequence_ids, (
        "Catalog sequence_id values must be exactly the enabled "
        "GenerationPlan.jobs sequence IDs, in plan order."
    )

    assert len(catalog_df) == len(plan.jobs), (
        "Catalog must contain exactly one row per enabled GenerationPlan job."
    )

    assert catalog_df["sample_id"].tolist() == list(range(len(plan.jobs))), (
        "Catalog sample_id values must be deterministic, sequential, "
        "and zero-based."
    )

    disabled_sequence_ids = set(plan.disabled_sequence_ids)
    catalog_sequence_ids = set(catalog_df["sequence_id"])

    assert catalog_sequence_ids.isdisjoint(disabled_sequence_ids), (
        "Disabled workbook rows must not become catalog samples. "
        f"Unexpected disabled catalog rows: "
        f"{sorted(catalog_sequence_ids & disabled_sequence_ids)}"
    )

    assert set(catalog_df["selected_status"]) == {"workbook_enabled"}, (
        "All M7.1 catalog rows must represent workbook-enabled jobs."
    )

    if "disabled_reported" in catalog_df.columns:
        assert catalog_df["disabled_reported"].tolist() == [
            False
        ] * len(catalog_df), (
            "Catalog rows are selected/enabled rows, so disabled_reported "
            "must be False for every selected catalog sample."
        )


@pytest.mark.milestone("M7.2")
@pytest.mark.proves(
    "Catalog membership stays workbook-defined; manifest reconciliation "
    "reports artifact readiness without image loading."
)
def test_m7_2_manifest_reconciliation_reports_artifact_readiness_without_changing_catalog_membership():
    """
    Manifest reconciliation reports artifact readiness without changing membership.

    Contract:
    - Reconciliation returns one row per loaded catalog job.
    - Sequence IDs and order match the workbook-defined catalog jobs.
    - Artifact readiness is reported via field_status.
    - Missing or incomplete artifacts do not drop catalog rows.
    - The default path does not open generated images.

    Notebook / protocol: M7.2
    """

    with as_file(VALIDATION_ROOT) as validation_root:
        workbook_path = (
            validation_root
            / "configs"
            / "m6"
            / "m6_matrix_plan.xlsx"
        )

        assert workbook_path.exists(), (
            "Installed validation workbook is missing: "
            f"{workbook_path}"
        )

        catalog_jobs = load_catalog_jobs(
            workbook_path=workbook_path,
            root_path=validation_root,
        )
        reconciliation_df = reconcile_catalog_jobs_with_manifest(catalog_jobs)

    expected_sequence_ids = [job.sequence_id for job in catalog_jobs]

    assert len(reconciliation_df) == len(catalog_jobs), (
        "Reconciliation must return exactly one row per catalog job."
    )
    assert reconciliation_df["sequence_id"].tolist() == expected_sequence_ids, (
        "Reconciliation sequence_id values must match catalog jobs in order."
    )
    assert reconciliation_df["sample_id"].tolist() == list(
        range(len(catalog_jobs))
    ), (
        "Reconciliation sample_id values must remain deterministic and "
        "zero-based."
    )

    missing_columns = [
        name
        for name in REQUIRED_RECONCILIATION_COLUMNS
        if name not in reconciliation_df.columns
    ]
    assert not missing_columns, (
        "Reconciliation dataframe is missing required columns: "
        f"{missing_columns}"
    )

    assert "deep_validation_ok" not in reconciliation_df.columns, (
        "Default reconciliation must not emit deep_validation_ok."
    )

    observed_statuses = set(reconciliation_df["field_status"].tolist())
    assert observed_statuses.issubset(ALLOWED_FIELD_STATUSES), (
        "Unexpected field_status values: "
        f"{sorted(observed_statuses - ALLOWED_FIELD_STATUSES)}"
    )


@pytest.mark.milestone("M7.3")
@pytest.mark.proves(
    "Acquisition order is part of observation identity; "
    "filtering yields a common shape without padding or masks."
)
def test_m7_3_schedule_consistent_subset_rejects_mixed_acquisition_shapes():
    """
    Schedule-consistent filtering yields one acquisition shape without padding.

    Contract:
    - The prototype enabled matrix catalog mixes acquisition schedules.
    - schedule_identity_table exposes ordered-angle schedule identity fields.
    - Filtering by camera_schedule_id yields one V, one ordered angle list,
      and one resolution.
    - angles_hash is sensitive to acquisition order, not a sorted bag of angles.
    - No images are opened.

    Notebook / protocol: M7.3
    """

    with as_file(VALIDATION_ROOT) as validation_root:
        workbook_path = (
            validation_root
            / "configs"
            / "m6"
            / "m6_matrix_plan.xlsx"
        )

        assert workbook_path.exists(), (
            "Installed validation workbook is missing: "
            f"{workbook_path}"
        )

        catalog_jobs = load_catalog_jobs(
            workbook_path=workbook_path,
            root_path=validation_root,
        )
        identity_df = schedule_identity_table(catalog_jobs)

    assert len(catalog_jobs) >= 2, (
        "Prototype workbook must provide multiple enabled catalog jobs."
    )

    missing_columns = [
        name
        for name in REQUIRED_SCHEDULE_IDENTITY_COLUMNS
        if name not in identity_df.columns
    ]
    assert not missing_columns, (
        "Schedule identity table is missing required columns: "
        f"{missing_columns}"
    )

    assert set(identity_df["schedule_status"]).issubset(
        ALLOWED_SCHEDULE_STATUSES
    ), (
        "Unexpected schedule_status values: "
        f"{sorted(set(identity_df['schedule_status']) - ALLOWED_SCHEDULE_STATUSES)}"
    )

    assert set(identity_df["schedule_status"]) == {"inconsistent"}, (
        "Full enabled matrix catalog must be marked schedule-inconsistent "
        "because it mixes acquisition shapes."
    )

    schedule_ids = {
        job.camera.camera_schedule_id for job in catalog_jobs
    }
    assert "orbit_matrix_006" in schedule_ids
    assert "orbit_matrix_012" in schedule_ids
    assert len(schedule_ids) > 1, (
        "Prototype workbook must mix multiple camera_schedule_id values."
    )

    subset_006 = filter_schedule_consistent(
        catalog_jobs,
        camera_schedule_id="orbit_matrix_006",
    )
    subset_012 = filter_schedule_consistent(
        catalog_jobs,
        camera_schedule_id="orbit_matrix_012",
    )

    assert [job.sequence_id for job in subset_006] == [
        "bear_m6_matrix_001"
    ]
    assert [job.sequence_id for job in subset_012] == [
        "bear_m6_matrix_002",
        "bear_m6_matrix_003",
    ]

    table_006 = schedule_identity_table(subset_006)
    table_012 = schedule_identity_table(subset_012)

    assert set(table_006["schedule_status"]) == {"consistent"}
    assert set(table_012["schedule_status"]) == {"consistent"}

    assert set(table_006["camera_schedule_id"]) == {"orbit_matrix_006"}
    assert set(table_006["frame_count"]) == {6}
    assert table_006["angles_hash"].nunique() == 1
    assert table_006[["resolution_x", "resolution_y"]].drop_duplicates().shape[
        0
    ] == 1

    assert set(table_012["camera_schedule_id"]) == {"orbit_matrix_012"}
    assert set(table_012["frame_count"]) == {12}
    assert table_012["angles_hash"].nunique() == 1
    assert table_012[["resolution_x", "resolution_y"]].drop_duplicates().shape[
        0
    ] == 1

    angles_006 = ordered_angles_deg(subset_006[0])
    assert angles_006[0] == table_006.loc[0, "first_angle_deg"]
    assert angles_006[-1] == table_006.loc[0, "last_angle_deg"]
    assert angles_hash(angles_006) == table_006.loc[0, "angles_hash"]
    assert angles_hash(angles_006) != angles_hash(tuple(reversed(angles_006))), (
        "angles_hash must use acquisition order, not a sorted angle bag."
    )


@pytest.mark.milestone("M7.4")
@pytest.mark.proves(
    "Catalog construction joins workbook jobs and optional manifests "
    "without treating samples as tensors."
)
def test_m7_4_flat_catalog_joins_jobs_and_manifests_without_loading_tensors():
    """
    Flat catalog construction joins workbook jobs and manifests without tensors.

    Contract:
    - One CatalogRow per workbook-defined catalog job, in job order.
    - Job label fields are filled from SequenceJob.
    - Manifest fields (schema_version, composition_domain, RoleRefs) are
      joined when artifacts are present.
    - field_status reports readiness using the reconciliation status vocabulary.
    - Samples remain catalog rows / RoleRefs, not image tensors.

    Notebook / protocol: M7.4
    """

    with as_file(VALIDATION_ROOT) as validation_root:
        workbook_path = (
            validation_root
            / "configs"
            / "m6"
            / "m6_matrix_plan.xlsx"
        )

        assert workbook_path.exists(), (
            "Installed validation workbook is missing: "
            f"{workbook_path}"
        )

        catalog_jobs = load_catalog_jobs(
            workbook_path=workbook_path,
            root_path=validation_root,
        )
        catalog_rows = build_catalog_rows(catalog_jobs)
        reconciliation_df = reconcile_catalog_jobs_with_manifest(catalog_jobs)

    assert len(catalog_rows) == len(catalog_jobs), (
        "Flat catalog must contain exactly one row per catalog job."
    )
    assert [row.sequence_id for row in catalog_rows] == [
        job.sequence_id for job in catalog_jobs
    ], (
        "Flat catalog sequence_id values must match catalog jobs in order."
    )
    assert [row.sample_id for row in catalog_rows] == list(
        range(len(catalog_jobs))
    ), (
        "Flat catalog sample_id values must be deterministic and zero-based."
    )

    assert all(isinstance(row, CatalogRow) for row in catalog_rows)

    observed_statuses = {row.field_status for row in catalog_rows}
    assert observed_statuses.issubset(ALLOWED_FIELD_STATUSES), (
        "Unexpected field_status values: "
        f"{sorted(observed_statuses - ALLOWED_FIELD_STATUSES)}"
    )

    for job, row, (_, recon) in zip(
        catalog_jobs,
        catalog_rows,
        reconciliation_df.iterrows(),
        strict=True,
    ):
        assert row.field_status == recon["field_status"]
        assert row.schema_version == recon["schema_version"]

        assert row.split == job.split
        assert row.camera_schedule_id == job.camera.camera_schedule_id
        assert row.frame_count == job.camera.num_views
        assert row.bear_mu_s == job.optical.mu_s
        assert row.bear_mu_a == job.optical.mu_a
        assert row.diffusion_setup_id == job.diffusion.diffusion_setup_id
        assert row.extrapolation_length == job.diffusion.extrapolation_length
        assert row.angles_deg == ordered_angles_deg(job)
        assert row.angles_hash == angles_hash(row.angles_deg)

        particle_present = job.particle.particle_kind != "none"
        assert row.particle_present is particle_present
        assert row.n_particles == len(job.particles)
        assert row.particle_group_id == job.particle_group_id
        assert len(row.particles) == len(job.particles)
        if particle_present and len(job.particles) == 1:
            assert row.particle_x == job.particle.center_x
            assert row.particle_y == job.particle.center_y
            assert row.particle_z == job.particle.center_z
            assert row.particle_radius == job.particle.radius
        elif particle_present:
            assert row.particle_x is None
            assert row.particles[0].center_x == job.particles[0].center_x
        else:
            assert row.particle_x is None

        for ref in (
            row.observed_ref,
            row.clean_ref,
            row.particle_ref,
            row.anomaly_ref,
        ):
            if ref is not None:
                assert isinstance(ref, RoleRef)
                assert not hasattr(ref, "shape"), (
                    "RoleRef must not look like an image tensor."
                )

        if row.field_status == "complete":
            assert row.schema_version is not None
            assert row.image_domain == "camera_intensity"
            assert row.composition_domain == (
                "linear_camera_intensity_before_jpeg"
            )
            assert row.observed_ref is not None
            assert row.clean_ref is not None
            assert row.particle_ref is not None
            assert row.anomaly_ref is not None
            assert row.anomaly_ref.role_name == "anomaly"

    # Schedule-consistent subset remains one CatalogRow per selected job.
    subset_012 = filter_schedule_consistent(
        catalog_jobs,
        camera_schedule_id="orbit_matrix_012",
    )
    subset_rows = build_catalog_rows(subset_012)
    assert len(subset_rows) == len(subset_012)
    assert [row.sequence_id for row in subset_rows] == [
        job.sequence_id for job in subset_012
    ]
    assert {row.field_status for row in subset_rows} == {"complete"}
    assert {row.frame_count for row in subset_rows} == {12}


@pytest.mark.milestone("M7.5")
@pytest.mark.proves(
    "Catalog rows convert to task-specific lazy (x, y) by selecting "
    "rows and fields — not by redefining the sample as a tensor."
)
def test_m7_5_task_dataset_returns_lazy_xy_without_redefining_sample_as_tensor():
    """
    Task datasets return lazy (x, y) field dicts without tensor catalog rows.

    Contract:
    - A DatasetTaskSpec selects rows and X/Y fields over CatalogRow objects.
    - build_task_dataset keeps selected CatalogRow objects (not tensors).
    - dataset[i] returns (x, y) field dicts for one index only.
    - RoleRef fields load lazily to numpy arrays with shape (V, C, H, W).
    - Validation pins image_representation=jpeg_uint8 (historical JPG path).
    - Scalar catalog fields are returned as ordinary values.
    - The same catalog supports distinct task views (inpainting vs localization).

    Notebook / protocol: M7.5
    """

    with as_file(VALIDATION_ROOT) as validation_root:
        workbook_path = (
            validation_root
            / "configs"
            / "m6"
            / "m6_matrix_plan.xlsx"
        )

        assert workbook_path.exists(), (
            "Installed validation workbook is missing: "
            f"{workbook_path}"
        )

        catalog_jobs = load_catalog_jobs(
            workbook_path=workbook_path,
            root_path=validation_root,
        )
        subset_012 = filter_schedule_consistent(
            catalog_jobs,
            camera_schedule_id="orbit_matrix_012",
        )
        catalog_rows = build_catalog_rows(subset_012)

    assert len(catalog_rows) >= 1
    assert all(isinstance(row, CatalogRow) for row in catalog_rows)
    assert {row.field_status for row in catalog_rows} == {"complete"}
    assert {row.frame_count for row in catalog_rows} == {12}

    inpainting_task = DatasetTaskSpec(
        name="inpainting",
        row_filter={"split": "train", "field_status": "complete"},
        x_fields=("observed_ref",),
        y_fields=("clean_ref",),
        image_representation=M7_IMAGE_REPRESENTATION,
    )
    localization_task = DatasetTaskSpec(
        name="localization",
        row_filter={"split": "train", "field_status": "complete"},
        x_fields=("particle_ref",),
        y_fields=(
            "particle_x",
            "particle_y",
            "particle_z",
            "particle_radius",
        ),
        image_representation=M7_IMAGE_REPRESENTATION,
    )

    dataset_inpainting = build_task_dataset(catalog_rows, inpainting_task)
    dataset_localization = build_task_dataset(catalog_rows, localization_task)

    assert len(dataset_inpainting) == len(catalog_rows)
    assert len(dataset_localization) == len(catalog_rows)
    assert all(isinstance(row, CatalogRow) for row in dataset_inpainting.rows)
    assert dataset_inpainting.task is inpainting_task
    assert dataset_localization.task is localization_task
    assert inpainting_task.image_representation == "jpeg_uint8"

    # Construction must not turn catalog rows into image tensors.
    for row in dataset_inpainting.rows:
        assert isinstance(row.observed_ref, RoleRef)
        assert isinstance(row.clean_ref, RoleRef)
        assert not hasattr(row.observed_ref, "shape")
        assert not hasattr(row.clean_ref, "shape")

    x_inpaint, y_inpaint = dataset_inpainting[0]
    assert set(x_inpaint) == {"observed_ref"}
    assert set(y_inpaint) == {"clean_ref"}

    observed = x_inpaint["observed_ref"]
    clean = y_inpaint["clean_ref"]
    assert isinstance(observed, np.ndarray)
    assert isinstance(clean, np.ndarray)
    assert observed.dtype == np.uint8
    assert clean.dtype == np.uint8
    assert observed.ndim == 4
    assert clean.ndim == 4
    assert observed.shape == clean.shape
    assert observed.shape[0] == 12  # V from orbit_matrix_012
    assert observed.shape[1] >= 1  # C (grayscale fixtures use C=1)
    assert observed.shape == (12, observed.shape[1], observed.shape[2], observed.shape[3])

    x_loc, y_loc = dataset_localization[0]
    assert set(x_loc) == {"particle_ref"}
    assert set(y_loc) == {
        "particle_x",
        "particle_y",
        "particle_z",
        "particle_radius",
    }
    particle = x_loc["particle_ref"]
    assert isinstance(particle, np.ndarray)
    assert particle.dtype == np.uint8
    assert particle.shape == observed.shape
    for key in localization_task.y_fields:
        assert isinstance(y_loc[key], float)
        assert y_loc[key] == getattr(dataset_localization.rows[0], key)

    # Row filter still excludes incomplete rows when asked.
    empty_task = DatasetTaskSpec(
        name="empty_incomplete",
        row_filter={"field_status": "incomplete_catalog"},
        x_fields=("observed_ref",),
        y_fields=("clean_ref",),
        image_representation=M7_IMAGE_REPRESENTATION,
    )
    empty_dataset = build_task_dataset(catalog_rows, empty_task)
    assert len(empty_dataset) == 0


@pytest.mark.milestone("M7.6")
@pytest.mark.proves(
    "The workbook-defined catalog plus simple subsetting is enough "
    "for later M8 task training interfaces."
)
def test_m7_6_workbook_catalog_and_subset_support_dual_task_views():
    """
    Workbook catalog plus schedule subsetting supports dual task training views.

    Contract:
    - Every workbook-selected job remains one catalog row with field_status.
    - A schedule-consistent subset can be marked without dropping other rows
      from the full catalog.
    - That subset supports both an inpainting view and a localization view.
    - For at least one complete index in each view, (x, y) returns valid
      numpy arrays / numeric values with role shape (V, C, H, W).

    Notebook / protocol: M7.6
    """

    schedule_id = "orbit_matrix_012"

    with as_file(VALIDATION_ROOT) as validation_root:
        workbook_path = (
            validation_root
            / "configs"
            / "m6"
            / "m6_matrix_plan.xlsx"
        )

        assert workbook_path.exists(), (
            "Installed validation workbook is missing: "
            f"{workbook_path}"
        )

        catalog_jobs = load_catalog_jobs(
            workbook_path=workbook_path,
            root_path=validation_root,
        )
        catalog_rows = build_catalog_rows(catalog_jobs)
        subset_jobs = filter_schedule_consistent(
            catalog_jobs,
            camera_schedule_id=schedule_id,
        )
        subset_rows = build_catalog_rows(subset_jobs)

    assert len(catalog_rows) == len(catalog_jobs)
    assert [row.sequence_id for row in catalog_rows] == [
        job.sequence_id for job in catalog_jobs
    ]
    assert all(isinstance(row, CatalogRow) for row in catalog_rows)
    assert {row.field_status for row in catalog_rows}.issubset(
        ALLOWED_FIELD_STATUSES
    )

    subset_ids = {row.sequence_id for row in subset_rows}
    assert subset_ids, "Schedule-consistent subset must be non-empty."
    assert subset_ids < {row.sequence_id for row in catalog_rows}, (
        "Prototype matrix must mix schedules so some workbook rows "
        "remain outside the chosen ML-ready subset."
    )
    assert {row.camera_schedule_id for row in subset_rows} == {schedule_id}
    assert {row.field_status for row in subset_rows} == {"complete"}

    # Notebook-facing catalog_status rollup vocabulary.
    for row in catalog_rows:
        in_subset = row.sequence_id in subset_ids
        if row.field_status != "complete":
            catalog_status = "incomplete_artifact"
        elif not in_subset:
            catalog_status = "excluded_by_schedule"
        else:
            catalog_status = "ready"
        assert catalog_status in {
            "ready",
            "incomplete_artifact",
            "excluded_by_schedule",
        }

    inpainting_task = DatasetTaskSpec(
        name="inpainting",
        row_filter={"split": "train", "field_status": "complete"},
        x_fields=("observed_ref",),
        y_fields=("clean_ref",),
        image_representation=M7_IMAGE_REPRESENTATION,
    )
    localization_task = DatasetTaskSpec(
        name="particle_localization",
        row_filter={"split": "train", "field_status": "complete"},
        x_fields=("observed_ref",),
        y_fields=(
            "particle_x",
            "particle_y",
            "particle_z",
            "particle_radius",
        ),
        image_representation=M7_IMAGE_REPRESENTATION,
    )

    dataset_inpainting = build_task_dataset(subset_rows, inpainting_task)
    dataset_localization = build_task_dataset(subset_rows, localization_task)

    assert len(dataset_inpainting) >= 1
    assert len(dataset_localization) >= 1

    x_inpaint, y_inpaint = dataset_inpainting[0]
    observed = x_inpaint["observed_ref"]
    clean = y_inpaint["clean_ref"]
    assert isinstance(observed, np.ndarray)
    assert isinstance(clean, np.ndarray)
    assert observed.dtype == np.uint8
    assert clean.dtype == np.uint8
    assert observed.shape == clean.shape
    assert observed.ndim == 4
    assert observed.shape[0] == subset_rows[0].frame_count

    x_loc, y_loc = dataset_localization[0]
    assert isinstance(x_loc["observed_ref"], np.ndarray)
    assert x_loc["observed_ref"].dtype == np.uint8
    assert x_loc["observed_ref"].shape == observed.shape
    for key in localization_task.y_fields:
        assert isinstance(y_loc[key], float)
        assert y_loc[key] == getattr(dataset_localization.rows[0], key)