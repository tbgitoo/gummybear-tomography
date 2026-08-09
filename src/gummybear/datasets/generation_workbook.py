"""Excel generation workbook parser and template writers.

The workbook is a human-facing control surface. Runtime semantics live in typed
configs from :mod:`gummybear.datasets.generation_plan`, not in Excel formulas.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REQUIRED_SHEETS: tuple[str, ...] = (
    "sequences",
    "optical_setups",
    "particles",
    "diffusion_setups",
    "camera_schedules",
    "corruptions",
)

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "sequences": (
        "sequence_id",
        "split",
        "seed",
        "phantom_id",
        "stl_path",
        "forward_model_tier",
        "optical_setup_id",
        "particle_setup_id",
        "diffusion_setup_id",
        "camera_schedule_id",
        "corruption_setup_id",
        "output_root",
        "enabled",
        "notes",
    ),
    "optical_setups": (
        "optical_setup_id",
        "illumination_kind",
        "light_position_x",
        "light_position_y",
        "light_position_z",
        "num_source_rays",
        "source_intensity",
        "mu_s",
        "mu_a",
        "refractive_index",
        "source_deposition_method",
        "cache_policy",
    ),
    "particles": (
        "particle_setup_id",
        "particle_kind",
        "center_x",
        "center_y",
        "center_z",
        "radius",
        "mu_s_particle",
        "mu_a_particle",
        "refractive_index_particle",
        "placement_mode",
        "seed",
        "enabled",
    ),
    "diffusion_setups": (
        "diffusion_setup_id",
        "g",
        "robin_boundary_model",
        "extrapolation_length",
        "fem_order",
        "solver_tolerance",
        "alpha_direct",
    ),
    "camera_schedules": (
        "camera_schedule_id",
        "schedule_kind",
        "num_views",
        "angle_start_deg",
        "angle_stop_deg",
        "axis_x",
        "axis_y",
        "axis_z",
        "resolution_x",
        "resolution_y",
        "camera_kind",
        "distance",
        "elevation_deg",
        "lateral_offsets",
        "z_offsets",
        "up_variants",
    ),
    "corruptions": (
        "corruption_setup_id",
        "corruption_kind",
        "amplitude",
        "frames",
        "seed",
        "composition_domain",
        "enabled",
    ),
}

SETUP_ID_COLUMNS: dict[str, str] = {
    "sequences": "sequence_id",
    "optical_setups": "optical_setup_id",
    "particles": "particle_setup_id",
    "diffusion_setups": "diffusion_setup_id",
    "camera_schedules": "camera_schedule_id",
    "corruptions": "corruption_setup_id",
}

OPENPYXL_INSTALL_HINT = (
    "openpyxl is required to read M6 generation workbooks. "
    "From the repository root, install it with:\n"
    "  pip install openpyxl -c requirements.txt"
)


class WorkbookValidationError(ValueError):
    """Raised when an M6 workbook cannot be parsed or is structurally invalid."""


@dataclass(frozen=True)
class WorkbookRow:
    """One normalized workbook row with sheet provenance.

    Attributes:
        sheet: Sheet name (e.g. ``sequences``, ``particles``).
        excel_row: 1-based Excel row number (header is row 1).
        values: Column name → normalized cell value.
        enabled: Whether the row participates in planning/generation.
    """

    sheet: str
    excel_row: int
    values: dict[str, Any]
    enabled: bool = True

    @property
    def setup_id(self) -> str | None:
        """Primary id column for this sheet, if defined."""
        column = SETUP_ID_COLUMNS.get(self.sheet)
        if column is None:
            return None
        value = self.values.get(column)
        if value is None:
            return None
        return str(value)


@dataclass(frozen=True)
class M6Workbook:
    """Parsed generation workbook (structure only; no physics).

    Excel is a control surface. Cross-sheet validation and cache identity live
    in :mod:`gummybear.datasets.generation_plan`.

    Attributes:
        path: Resolved workbook path.
        sha256: Content hash of workbook bytes.
        sheets: Normalized rows keyed by required sheet name.
        sheet_names: Ordered required sheet names present.
        warnings: Non-fatal parse notices.
    """

    path: Path
    sha256: str
    sheets: dict[str, tuple[WorkbookRow, ...]]
    sheet_names: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def rows(self, sheet: str) -> tuple[WorkbookRow, ...]:
        """Return all rows for ``sheet`` (raises if unknown)."""
        if sheet not in self.sheets:
            raise WorkbookValidationError(f"Unknown sheet: {sheet!r}")
        return self.sheets[sheet]

    def enabled_rows(self, sheet: str) -> tuple[WorkbookRow, ...]:
        """Return workbook rows marked for execution on ``sheet``.

        Filters :meth:`rows` to those with ``enabled=True``. Disabled rows stay
        in the workbook but are excluded from planning and generation.
        """
        return tuple(row for row in self.rows(sheet) if row.enabled)

    def disabled_rows(self, sheet: str) -> tuple[WorkbookRow, ...]:
        """Return rows with ``enabled=False``."""
        return tuple(row for row in self.rows(sheet) if not row.enabled)


def _require_pandas_openpyxl():
    try:
        import pandas as pd
    except ImportError as exc:
        raise WorkbookValidationError(
            "pandas is required to read M6 generation workbooks."
        ) from exc

    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise WorkbookValidationError(OPENPYXL_INSTALL_HINT) from exc

    return pd


def workbook_sha256(path: Path | str) -> str:
    """Return the SHA256 digest of the workbook file bytes."""
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def normalize_enabled(value: Any, *, default: bool = True) -> bool:
    """Normalize common Excel boolean/enabled representations."""
    if _is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    raise WorkbookValidationError(f"Cannot normalize enabled value: {value!r}")


def normalize_cell(value: Any) -> Any:
    """Normalize empty / none-like Excel cells."""
    if _is_missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        if text.lower() in {"none", "null", "nan"}:
            return "none" if text.lower() == "none" else None
        return text
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _row_context(sheet: str, excel_row: int, values: Mapping[str, Any]) -> str:
    id_column = SETUP_ID_COLUMNS.get(sheet)
    setup_id = None
    if id_column is not None:
        setup_id = values.get(id_column)
    if setup_id is not None and not _is_missing(setup_id):
        return f"sheet={sheet!r} excel_row={excel_row} {id_column}={setup_id!r}"
    return f"sheet={sheet!r} excel_row={excel_row}"


def _normalize_sheet_dataframe(sheet: str, df) -> tuple[WorkbookRow, ...]:
    required = REQUIRED_COLUMNS[sheet]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise WorkbookValidationError(
            f"Sheet {sheet!r} is missing required columns: {missing}"
        )

    rows: list[WorkbookRow] = []
    for offset, (_, series) in enumerate(df.iterrows()):
        excel_row = offset + 2  # header is row 1
        values = {column: normalize_cell(series[column]) for column in required}

        # Preserve optional extra columns for forward compatibility.
        for column in df.columns:
            key = str(column)
            if key not in values:
                values[key] = normalize_cell(series[column])

        id_column = SETUP_ID_COLUMNS[sheet]
        if values.get(id_column) is None:
            # Completely empty trailing rows are ignored.
            if all(values.get(column) is None for column in required):
                continue
            raise WorkbookValidationError(
                f"Missing {id_column} in {_row_context(sheet, excel_row, values)}"
            )

        enabled_default = True
        if sheet == "corruptions" and str(values.get(id_column)).lower() == "none":
            enabled_default = False

        if "enabled" in values:
            try:
                enabled = normalize_enabled(
                    values["enabled"],
                    default=enabled_default,
                )
            except WorkbookValidationError as exc:
                raise WorkbookValidationError(
                    f"{exc} ({_row_context(sheet, excel_row, values)})"
                ) from exc
            values["enabled"] = enabled
        else:
            enabled = enabled_default

        rows.append(
            WorkbookRow(
                sheet=sheet,
                excel_row=excel_row,
                values=values,
                enabled=enabled,
            )
        )

    return tuple(rows)


def load_generation_workbook(path: Path | str) -> M6Workbook:
    """Load and lightly normalize a multi-sheet generation workbook.

    Parses required sheets and normalizes cell values; does not run physics or
    fully validate cross-references. Call
    :func:`gummybear.datasets.generation_plan.validate_generation_plan` for
    scientific and planning validation.

    Notebook / protocol:
        M6 workbook control surface.
    """
    pd = _require_pandas_openpyxl()
    path = Path(path).resolve()
    if not path.is_file():
        raise WorkbookValidationError(f"Workbook not found: {path}")

    try:
        raw_sheets = pd.read_excel(
            path,
            sheet_name=list(REQUIRED_SHEETS),
            engine="openpyxl",
        )
    except ValueError as exc:
        # pandas raises ValueError when requested sheets are missing.
        raise WorkbookValidationError(
            f"Failed to read required sheets from {path}: {exc}"
        ) from exc
    except ImportError as exc:
        raise WorkbookValidationError(OPENPYXL_INSTALL_HINT) from exc

    sheets: dict[str, tuple[WorkbookRow, ...]] = {}
    warnings: list[str] = []
    for sheet_name in REQUIRED_SHEETS:
        df = raw_sheets[sheet_name]
        sheets[sheet_name] = _normalize_sheet_dataframe(sheet_name, df)

    digest = workbook_sha256(path)
    return M6Workbook(
        path=path,
        sha256=digest,
        sheets=sheets,
        sheet_names=tuple(REQUIRED_SHEETS),
        warnings=tuple(warnings),
    )


def example_workbook_frames() -> dict[str, "Any"]:
    """Return pandas DataFrames for the Phase 1 smoke workbook template."""
    pd = _require_pandas_openpyxl()

    sequences = pd.DataFrame(
        [
            {
                "sequence_id": "bear_m6_smoke_001",
                "split": "train",
                "seed": 42,
                "phantom_id": "proto_bear",
                "stl_path": "cad/proto_bear.stl",
                "forward_model_tier": ("m5_refractive_diffusion_particle_perturbation"),
                "optical_setup_id": "opt_smoke_backlight_001",
                "particle_setup_id": "particle_smoke_sphere_001",
                "diffusion_setup_id": "diff_smoke_robin_001",
                "camera_schedule_id": "orbit_smoke_006",
                "corruption_setup_id": "none",
                "output_root": "data/generated/m6_2",
                "enabled": True,
                "notes": "Phase 1 dry-run smoke plan",
            }
        ]
    )

    optical_setups = pd.DataFrame(
        [
            {
                "optical_setup_id": "opt_smoke_backlight_001",
                "illumination_kind": "point",
                "light_position_x": 15.0,
                "light_position_y": 15.0,
                "light_position_z": 40.0,
                "num_source_rays": 512,
                "source_intensity": 1.0,
                "mu_s": 0.3,
                "mu_a": 0.1,
                "refractive_index": 1.33,
                "source_deposition_method": "exact_ray_tet_intervals",
                "cache_policy": "reuse",
            }
        ]
    )

    # Validated M5D smoke placement: canonical diffusion-mesh bounds center.
    # This is fixture provenance, not an architecture-wide fixed coordinate.
    particles = pd.DataFrame(
        [
            {
                "particle_setup_id": "particle_smoke_sphere_001",
                "particle_kind": "sphere",
                "center_x": -1.1180591583251953,
                "center_y": 0.4537315368652344,
                "center_z": 2.5,
                "radius": 3.0,
                "mu_s_particle": 0.8,
                "mu_a_particle": 0.2,
                "refractive_index_particle": 1.33,
                "placement_mode": "fixed",
                "seed": 42,
                "enabled": True,
            }
        ]
    )

    diffusion_setups = pd.DataFrame(
        [
            {
                "diffusion_setup_id": "diff_smoke_robin_001",
                "g": 0.0,
                "robin_boundary_model": "effective_refractive_boundary",
                "extrapolation_length": 5.0,
                "fem_order": 1,
                "solver_tolerance": 1.0e-8,
                "alpha_direct": 0.0,
            }
        ]
    )

    camera_schedules = pd.DataFrame(
        [
            {
                "camera_schedule_id": "orbit_smoke_006",
                "schedule_kind": "orbit",
                "num_views": 6,
                "angle_start_deg": 0.0,
                "angle_stop_deg": 300.0,
                "axis_x": 0.0,
                "axis_y": 0.0,
                "axis_z": 1.0,
                "resolution_x": 128,
                "resolution_y": 128,
                "camera_kind": "orbit",
                "distance": 80.0,
                "elevation_deg": 0.0,
                "lateral_offsets": "none",
                "z_offsets": "none",
                "up_variants": "none",
            }
        ]
    )

    corruptions = pd.DataFrame(
        [
            {
                "corruption_setup_id": "none",
                "corruption_kind": "none",
                "amplitude": 0.0,
                "frames": "none",
                "seed": 0,
                "composition_domain": "none",
                "enabled": False,
            }
        ]
    )

    return {
        "sequences": sequences,
        "optical_setups": optical_setups,
        "particles": particles,
        "diffusion_setups": diffusion_setups,
        "camera_schedules": camera_schedules,
        "corruptions": corruptions,
    }


def write_example_generation_workbook(path: Path | str) -> Path:
    """Write the Phase 1 smoke workbook template to ``path``."""
    pd = _require_pandas_openpyxl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = example_workbook_frames()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in REQUIRED_SHEETS:
            frames[sheet_name].to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )
    return path.resolve()


def matrix_workbook_frames() -> dict[str, "Any"]:
    """Return the dedicated M6.5 camera/diffusion matrix workbook frames."""
    pd = _require_pandas_openpyxl()
    frames = example_workbook_frames()

    optical = frames["optical_setups"].copy()
    optical.loc[0, "optical_setup_id"] = "opt_matrix_backlight_001"

    particles = frames["particles"].copy()
    particles.loc[0, "particle_setup_id"] = "particle_matrix_sphere_001"

    diffusion = pd.concat(
        [frames["diffusion_setups"], frames["diffusion_setups"]],
        ignore_index=True,
    )
    diffusion.loc[0, "diffusion_setup_id"] = "diff_matrix_robin_l05"
    diffusion.loc[0, "extrapolation_length"] = 5.0
    diffusion.loc[1, "diffusion_setup_id"] = "diff_matrix_robin_l12"
    diffusion.loc[1, "extrapolation_length"] = 12.0

    cameras = pd.concat(
        [frames["camera_schedules"], frames["camera_schedules"]],
        ignore_index=True,
    )
    cameras.loc[0, "camera_schedule_id"] = "orbit_matrix_006"
    cameras.loc[0, "num_views"] = 6
    cameras.loc[0, "angle_stop_deg"] = 300.0
    cameras.loc[1, "camera_schedule_id"] = "orbit_matrix_012"
    cameras.loc[1, "num_views"] = 12
    cameras.loc[1, "angle_stop_deg"] = 330.0

    shared = {
        "split": "train",
        "seed": 42,
        "phantom_id": "proto_bear",
        "stl_path": "cad/proto_bear.stl",
        "forward_model_tier": ("m5_refractive_diffusion_particle_perturbation"),
        "optical_setup_id": "opt_matrix_backlight_001",
        "particle_setup_id": "particle_matrix_sphere_001",
        "corruption_setup_id": "none",
        "output_root": "data/generated/m6_5",
    }
    sequences = pd.DataFrame(
        [
            {
                **shared,
                "sequence_id": "bear_m6_matrix_001",
                "diffusion_setup_id": "diff_matrix_robin_l05",
                "camera_schedule_id": "orbit_matrix_006",
                "enabled": True,
                "notes": "M6.5 baseline matrix sequence",
            },
            {
                **shared,
                "sequence_id": "bear_m6_matrix_002",
                "diffusion_setup_id": "diff_matrix_robin_l05",
                "camera_schedule_id": "orbit_matrix_012",
                "enabled": True,
                "notes": "M6.5 camera schedule variant",
            },
            {
                **shared,
                "sequence_id": "bear_m6_matrix_003",
                "diffusion_setup_id": "diff_matrix_robin_l12",
                "camera_schedule_id": "orbit_matrix_012",
                "enabled": True,
                "notes": "M6.5 diffusion variant",
            },
            {
                **shared,
                "sequence_id": "bear_m6_matrix_004",
                "diffusion_setup_id": "diff_matrix_robin_l12",
                "camera_schedule_id": "orbit_matrix_006",
                "enabled": False,
                "notes": "Prepared disabled row for delta-generation demonstration",
            },
        ]
    )

    return {
        "sequences": sequences,
        "optical_setups": optical,
        "particles": particles,
        "diffusion_setups": diffusion,
        "camera_schedules": cameras,
        "corruptions": frames["corruptions"].copy(),
    }


def write_matrix_generation_workbook(path: Path | str) -> Path:
    """Write the dedicated M6.5 matrix workbook to ``path``."""
    pd = _require_pandas_openpyxl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = matrix_workbook_frames()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in REQUIRED_SHEETS:
            frames[sheet_name].to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )
    return path.resolve()


# Default offline two-sphere layout for the checked-in multi-particle workbook.
# Centres are far enough apart that ``ParticleSet.validate()`` accepts them.
DEFAULT_MULTI_PARTICLE_CENTERS: tuple[tuple[float, float, float], ...] = (
    (-5.0, 0.5, 2.5),
    (5.0, -0.5, 2.5),
)
DEFAULT_MULTI_PARTICLE_GROUP_ID = "dryrun_two_sphere"
DEFAULT_MULTI_PARTICLE_SEQUENCE_ID = "bear_m6_multi_001"
DEFAULT_MULTI_PARTICLE_RADIUS = 3.0


def multi_particle_workbook_frames(
    *,
    centers: list[tuple[float, float, float]]
    | tuple[tuple[float, float, float], ...]
    | None = None,
    sequence_id: str = DEFAULT_MULTI_PARTICLE_SEQUENCE_ID,
    particle_group_id: str = DEFAULT_MULTI_PARTICLE_GROUP_ID,
    radius: float = DEFAULT_MULTI_PARTICLE_RADIUS,
    output_root: str = "data/generated/m6_1",
) -> dict[str, "Any"]:
    """Return frames for a fixed two-or-more-particle dry-run workbook.

    Builds on the smoke template, then attaches an ordered non-overlapping
    particle group via :func:`attach_particle_group`. Runtime placement stays
    ``fixed``; centres are offline-authored.
    """
    frames = example_workbook_frames()
    sequences = frames["sequences"].copy()
    sequences.loc[0, "sequence_id"] = sequence_id
    sequences.loc[0, "output_root"] = output_root
    sequences.loc[0, "notes"] = (
        "Multi-particle dry-run / planning workbook "
        f"(group={particle_group_id})"
    )
    frames["sequences"] = sequences

    return attach_particle_group(
        frames,
        sequence_id=sequence_id,
        particle_group_id=particle_group_id,
        centers=centers if centers is not None else DEFAULT_MULTI_PARTICLE_CENTERS,
        radius=radius,
    )


def write_multi_particle_generation_workbook(path: Path | str) -> Path:
    """Write the checked-in multi-particle planning workbook to ``path``."""
    return write_generation_workbook_frames(path, multi_particle_workbook_frames())


def particle_setup_row(
    *,
    particle_setup_id: str,
    center: tuple[float, float, float],
    radius: float = 3.0,
    mu_s_particle: float = 0.8,
    mu_a_particle: float = 0.2,
    refractive_index_particle: float = 1.33,
    placement_mode: str = "fixed",
    seed: int | None = 42,
    enabled: bool = True,
    particle_group_id: str | None = None,
) -> dict[str, Any]:
    """One fixed-sphere particle sheet row (optional ``particle_group_id``)."""
    row: dict[str, Any] = {
        "particle_setup_id": particle_setup_id,
        "particle_kind": "sphere",
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "radius": float(radius),
        "mu_s_particle": float(mu_s_particle),
        "mu_a_particle": float(mu_a_particle),
        "refractive_index_particle": float(refractive_index_particle),
        "placement_mode": placement_mode,
        "seed": seed,
        "enabled": enabled,
    }
    if particle_group_id is not None:
        row["particle_group_id"] = particle_group_id
    return row


def attach_particle_group(
    frames: dict[str, Any],
    *,
    sequence_id: str,
    particle_group_id: str,
    centers: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
    radius: float = 3.0,
    particle_id_prefix: str | None = None,
    require_non_overlapping: bool = True,
) -> dict[str, Any]:
    """Return workbook frames with an ordered multi-particle group on one sequence.

    Centres are offline-authored fixed placements. Runtime generation stays
    ``placement_mode=fixed``. Workbook row order of the new particle rows is
    the scientific particle order.

    When ``require_non_overlapping`` is True (default), overlapping spheres
    raise ``ParticleOverlapError`` before the frames are returned.
    """
    pd = _require_pandas_openpyxl()
    if not centers:
        raise ValueError("attach_particle_group requires at least one centre.")

    if require_non_overlapping:
        from gummybear.particles import ParticleSet, ParticleSphere

        ParticleSet.from_particles(
            [
                ParticleSphere(
                    center=center,
                    radius=radius,
                    particle_id=f"{particle_id_prefix or particle_group_id}_p{index:03d}",
                )
                for index, center in enumerate(centers)
            ]
        )

    prefix = particle_id_prefix or particle_group_id
    new_rows = [
        particle_setup_row(
            particle_setup_id=f"{prefix}_p{index:03d}",
            center=center,
            radius=radius,
            particle_group_id=particle_group_id,
        )
        for index, center in enumerate(centers)
    ]
    particles = frames["particles"].copy()
    # Optional column may be absent on legacy frames.
    if "particle_group_id" not in particles.columns:
        particles["particle_group_id"] = None
    particles = pd.concat(
        [particles, pd.DataFrame(new_rows)],
        ignore_index=True,
    )

    sequences = frames["sequences"].copy()
    if "particle_group_id" not in sequences.columns:
        sequences["particle_group_id"] = None
    mask = sequences["sequence_id"] == sequence_id
    if not bool(mask.any()):
        raise ValueError(f"sequence_id={sequence_id!r} not found in frames.")
    sequences.loc[mask, "particle_group_id"] = particle_group_id
    sequences.loc[mask, "particle_setup_id"] = new_rows[0]["particle_setup_id"]

    updated = dict(frames)
    updated["particles"] = particles
    updated["sequences"] = sequences
    return updated


def write_generation_workbook_frames(
    path: Path | str,
    frames: dict[str, Any],
) -> Path:
    """Write arbitrary M6 sheet frames to an Excel workbook."""
    pd = _require_pandas_openpyxl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in REQUIRED_SHEETS:
            if sheet_name not in frames:
                raise WorkbookValidationError(
                    f"frames missing required sheet {sheet_name!r}"
                )
            frames[sheet_name].to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )
    return path.resolve()
