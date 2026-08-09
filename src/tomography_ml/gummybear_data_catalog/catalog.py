"""Flat catalog rows joined from workbook jobs and optional manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gummybear.datasets.generation_plan import ParticleSetupConfig, SequenceJob
from gummybear.datasets.output_plan import resolve_output_root

from .gummybear_adapter import (
    FIELD_STATUS_COMPLETE,
    FIELD_STATUS_DIRECTORY_MISSING,
    FIELD_STATUS_INCOMPLETE_CATALOG,
    FIELD_STATUS_MANIFEST_INVALID,
    FIELD_STATUS_MANIFEST_MISSING,
    FIELD_STATUS_STALE_JOB_HASH,
    angles_hash,
    manifest_resolved_job_hash_matches,
    ordered_angles_deg,
)


@dataclass(frozen=True)
class RoleRef:
    """Lazy handle to one multi-view role in a generated sequence.

    Stores only ``manifest_path`` and ``role_name``; pixel data loads on demand
    via :func:`~tomography_ml.gummybear_data_catalog.task_dataset.load_role_array`.
    Role names follow manifest conventions (``observed``, ``clean``,
    ``particle``, ``anomaly``). A role is **not** a single camera view — one
    loaded role tensor stacks all ``V`` acquisition angles.

    Attributes:
        manifest_path: Path to the sequence ``manifest.json``.
        role_name: Manifest role key to resolve under ``frames[].filenames``.
    """

    manifest_path: str
    role_name: str


@dataclass(frozen=True)
class ParticleLabel:
    """One particle localisation label from a workbook particle setup.

    Attributes:
        particle_setup_id: Workbook particle setup identifier.
        center_x: Sphere center x in phantom coordinates.
        center_y: Sphere center y.
        center_z: Sphere center z.
        radius: Sphere radius.
        mu_s: Particle scattering coefficient; ``None`` when unset.
        mu_a: Particle absorption coefficient; ``None`` when unset.
    """

    particle_setup_id: str
    center_x: float
    center_y: float
    center_z: float
    radius: float
    mu_s: float | None = None
    mu_a: float | None = None


@dataclass(frozen=True)
class CatalogRow:
    """One flat catalog sample joining a workbook job with optional manifest.

    A **sample** is one catalog row (one ``sequence_id``), not one camera view.
    Multi-view tensors load lazily via ``RoleRef`` fields; scalar labels come
    from the workbook. ``field_status`` reports manifest readiness without
    opening image files.

    Attributes:
        sample_id: Enumerated index within the catalog build (0-based).
        sequence_id: Canonical sequence identifier.
        split: Train / validation / test split from the workbook.
        output_root: Scenario output root from the job.
        sequence_dir: Resolved on-disk sequence directory.
        manifest_path: Path to ``manifest.json`` under ``sequence_dir``.
        field_status: On-disk readiness (``complete``, ``directory_missing``,
            ``stale_job_hash`` when the manifest ``resolved_job_hash`` does not
            exactly match the workbook job). Rows stay in the catalog; only
            ``complete`` should be used for training filters. Legacy manifest
            ``split`` / ``seed`` fields, if present, are ignored for identity.
        schema_version: Manifest schema version when readable; else ``None``.
        resolved_job_hash: Stable hash of the resolved workbook job.
        camera_schedule_id: Camera acquisition schedule identifier.
        frame_count: Number of views ``V`` in the sequence.
        angles_deg: Acquisition-order camera angles (not sorted).
        angles_hash: Stable hash of ``angles_deg`` for schedule joins.
        observed_ref: Lazy handle to the observed role; ``None`` if unavailable.
        clean_ref: Lazy handle to the clean role.
        particle_ref: Lazy handle to the particle role.
        anomaly_ref: Lazy handle to the anomaly role.
        optical_setup_id: Illumination / optical setup identifier.
        bear_mu_s: Phantom scattering coefficient from optical setup.
        bear_mu_a: Phantom absorption coefficient from optical setup.
        particle_present: ``True`` when at least one particle is configured.
        n_particles: Count of particles in the ordered particle group.
        particle_group_id: Workbook particle group identifier.
        particles: Tuple of per-particle localisation labels.
        particle_x: First-particle x when ``n_particles == 1``; else ``None``.
        particle_y: First-particle y when ``n_particles == 1``; else ``None``.
        particle_z: First-particle z when ``n_particles == 1``; else ``None``.
        particle_radius: First-particle radius when ``n_particles == 1``.
        particle_mu_s: First-particle scattering when ``n_particles == 1``.
        particle_mu_a: First-particle absorption when ``n_particles == 1``.
        diffusion_setup_id: Diffusion setup identifier.
        extrapolation_length: Robin extrapolation length from diffusion setup.
        image_domain: Manifest image domain (default ``camera_intensity``).
        composition_domain: Manifest composition domain when present.

    See also:
        :func:`build_catalog_row` — row construction from a :class:`~gummybear.datasets.generation_plan.SequenceJob`.
        :func:`~tomography_ml.gummybear_data_catalog.task_dataset.build_task_dataset` — ML task datasets over rows.
    """

    # identity
    sample_id: int
    sequence_id: str
    split: str

    # locations
    output_root: str
    sequence_dir: str
    manifest_path: str

    # status
    field_status: str
    schema_version: str | None

    # provenance
    resolved_job_hash: str

    # acquisition
    camera_schedule_id: str
    frame_count: int
    angles_deg: tuple[float, ...]
    angles_hash: str

    # role references
    observed_ref: RoleRef | None
    clean_ref: RoleRef | None
    particle_ref: RoleRef | None
    anomaly_ref: RoleRef | None

    # optical labels
    optical_setup_id: str
    bear_mu_s: float
    bear_mu_a: float

    # particle labels
    particle_present: bool
    n_particles: int
    particle_group_id: str
    particles: tuple[ParticleLabel, ...]

    # compatibility scalars: first particle only when n_particles == 1
    particle_x: float | None
    particle_y: float | None
    particle_z: float | None

    particle_radius: float | None

    particle_mu_s: float | None
    particle_mu_a: float | None

    # diffusion labels
    diffusion_setup_id: str
    extrapolation_length: float

    # representation metadata
    image_domain: str
    composition_domain: str | None


def compute_angles_hash(angles_deg: tuple[float, ...]) -> str:
    """Return a stable hash of acquisition-order camera angles.

    Backward-compatible alias of :func:`~tomography_ml.gummybear_data_catalog.gummybear_adapter.angles_hash`.
    Order is significant; do not sort ``angles_deg`` before calling.

    Args:
        angles_deg: Camera angles in frame / pose order.

    Returns:
        Lowercase hex sha256 digest.
    """
    return angles_hash(angles_deg)


def particle_label_from_setup(setup: ParticleSetupConfig) -> ParticleLabel:
    """Convert one workbook ``ParticleSetupConfig`` into a ``ParticleLabel``.

    Copies geometric center, radius, and optical coefficients into the flat
    catalog label type. Does not read manifests or validate on-disk artifacts.

    Args:
        setup: Single particle entry from a ``SequenceJob.particles`` group.

    Returns:
        Frozen label with workbook setup id and localisation scalars.
    """
    return ParticleLabel(
        particle_setup_id=setup.particle_setup_id,
        center_x=float(setup.center_x),
        center_y=float(setup.center_y),
        center_z=float(setup.center_z),
        radius=float(setup.radius),
        mu_s=float(setup.mu_s_particle),
        mu_a=float(setup.mu_a_particle),
    )


def build_catalog_rows(catalog_jobs) -> list[CatalogRow]:
    """Build one flat ``CatalogRow`` per workbook-defined catalog job.

    ``sample_id`` is the enumerate index over ``catalog_jobs``. Each row joins
    workbook fields with optional on-disk ``manifest.json`` metadata; image
    files are not opened. Rows with missing or invalid manifests remain in the
    list with a non-``complete`` ``field_status``.

    Args:
        catalog_jobs: Iterable of validated ``SequenceJob`` objects (typically
            from :func:`~tomography_ml.gummybear_data_catalog.gummybear_adapter.load_catalog_jobs`).

    Returns:
        List of :class:`CatalogRow` in the same order as ``catalog_jobs``.
    """
    return [
        build_catalog_row(sample_id=sample_id, job=job)
        for sample_id, job in enumerate(catalog_jobs)
    ]


def _role_available(payload: dict[str, Any], role_name: str) -> bool:
    """Return True if ``role_name`` is claimed in roles or frame filenames."""
    roles = payload.get("roles", {})
    if isinstance(roles, dict):
        if role_name in roles:
            return True
        # Manifest may alias anomaly_preview -> anomaly directory/filename key.
        if role_name in {str(value) for value in roles.values()}:
            return True
        if role_name == "anomaly" and "anomaly_preview" in roles:
            return True

    frames = payload.get("frames", [])
    if isinstance(frames, list) and frames:
        first = frames[0]
        if isinstance(first, dict):
            filenames = first.get("filenames", {})
            if isinstance(filenames, dict) and role_name in filenames:
                return True
    return False


def _optional_role_ref(
    *,
    manifest_path_display: str,
    payload: dict[str, Any] | None,
    role_name: str,
) -> RoleRef | None:
    if payload is None or not _role_available(payload, role_name):
        return None
    return RoleRef(manifest_path=manifest_path_display, role_name=role_name)


def build_catalog_row(*, sample_id: int, job: SequenceJob) -> CatalogRow:
    """Join one ``SequenceJob`` with optional on-disk manifest metadata.

    Resolves ``sequence_dir`` and reads ``manifest.json`` when present to set
    ``field_status``, ``schema_version``, role :class:`RoleRef` handles, and
    representation domains. Workbook fields (angles, labels, setup ids) always
    come from ``job``; missing manifests do not raise. A readable manifest is
    ``complete`` when ``sequence_id`` / ``schema_version`` look valid and the
    manifest ``resolved_job_hash`` exactly matches the workbook job. Legacy
    top-level ``split`` / ``seed`` on the manifest are ignored.

    Args:
        sample_id: Catalog enumerate index for this job.
        job: One validated workbook sequence job.

    Returns:
        Frozen :class:`CatalogRow` with lazy role refs when the manifest
        claims the role.

    See also:
        :func:`build_catalog_rows` — batch over workbook jobs.
        :func:`~tomography_ml.gummybear_data_catalog.task_dataset.build_task_dataset` — downstream ML dataset.
    """
    sequence_dir = resolve_output_root(job) / job.sequence_id
    manifest_path = sequence_dir / "manifest.json"
    sequence_dir_display = str(sequence_dir)
    manifest_path_display = str(manifest_path)

    schema_version: str | None = None
    payload: dict[str, Any] | None = None
    image_domain = "camera_intensity"
    composition_domain = job.corruption.composition_domain

    if not sequence_dir.is_dir():
        field_status = FIELD_STATUS_DIRECTORY_MISSING
    elif not manifest_path.is_file():
        field_status = FIELD_STATUS_MANIFEST_MISSING
    else:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            field_status = FIELD_STATUS_MANIFEST_INVALID
        else:
            if not isinstance(loaded, dict):
                field_status = FIELD_STATUS_MANIFEST_INVALID
            else:
                payload = loaded
                raw_sequence_id = payload.get("sequence_id")
                manifest_sequence_id = (
                    str(raw_sequence_id) if raw_sequence_id is not None else None
                )
                raw_schema = payload.get("schema_version")
                if raw_schema is not None:
                    schema_version = str(raw_schema)

                representation = payload.get("representation", {})
                if isinstance(representation, dict):
                    raw_image_domain = representation.get("image_domain")
                    if raw_image_domain is not None:
                        image_domain = str(raw_image_domain)
                    raw_composition = representation.get("composition_domain")
                    if raw_composition is not None:
                        composition_domain = str(raw_composition)

                if (
                    manifest_sequence_id != job.sequence_id
                    or schema_version is None
                ):
                    field_status = FIELD_STATUS_INCOMPLETE_CATALOG
                elif manifest_resolved_job_hash_matches(job, payload):
                    field_status = FIELD_STATUS_COMPLETE
                else:
                    field_status = FIELD_STATUS_STALE_JOB_HASH

    angles = ordered_angles_deg(job)
    particle_labels = tuple(
        particle_label_from_setup(item) for item in job.particles
    )
    n_particles = len(particle_labels)
    particle_present = n_particles > 0 and job.particle.particle_kind != "none"
    single = particle_present and n_particles == 1
    primary = job.particle if single else None

    return CatalogRow(
        sample_id=sample_id,
        sequence_id=job.sequence_id,
        split=job.split,
        output_root=str(job.output_root),
        sequence_dir=sequence_dir_display,
        manifest_path=manifest_path_display,
        field_status=field_status,
        schema_version=schema_version,
        resolved_job_hash=job.resolved_job_hash,
        camera_schedule_id=job.camera.camera_schedule_id,
        frame_count=int(job.camera.num_views),
        angles_deg=angles,
        angles_hash=angles_hash(angles),
        observed_ref=_optional_role_ref(
            manifest_path_display=manifest_path_display,
            payload=payload,
            role_name="observed",
        ),
        clean_ref=_optional_role_ref(
            manifest_path_display=manifest_path_display,
            payload=payload,
            role_name="clean",
        ),
        particle_ref=_optional_role_ref(
            manifest_path_display=manifest_path_display,
            payload=payload,
            role_name="particle",
        ),
        anomaly_ref=_optional_role_ref(
            manifest_path_display=manifest_path_display,
            payload=payload,
            role_name="anomaly",
        ),
        optical_setup_id=str(job.optical.optical_setup_id),
        bear_mu_s=float(job.optical.mu_s),
        bear_mu_a=float(job.optical.mu_a),
        particle_present=particle_present,
        n_particles=n_particles,
        particle_group_id=str(job.particle_group_id),
        particles=particle_labels,
        particle_x=(float(primary.center_x) if primary is not None else None),
        particle_y=(float(primary.center_y) if primary is not None else None),
        particle_z=(float(primary.center_z) if primary is not None else None),
        particle_radius=(float(primary.radius) if primary is not None else None),
        particle_mu_s=(
            float(primary.mu_s_particle) if primary is not None else None
        ),
        particle_mu_a=(
            float(primary.mu_a_particle) if primary is not None else None
        ),
        diffusion_setup_id=str(job.diffusion.diffusion_setup_id),
        extrapolation_length=float(job.diffusion.extrapolation_length),
        image_domain=image_domain,
        composition_domain=composition_domain,
    )
