"""Artifact-facing validation for generated multi-view sequence directories.

Validation is independent of finite-element method (FEM)/runtime physics. The authoritative anomaly
identity remains pre-JPEG; current camera display JPEG preview (lossy uint8) artifacts do not persist those
arrays, so that exact check is reported as not performed rather than inferred
from compressed role images.

Notebook / protocol: M6.3
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

from gummybear.datasets.manifest_writer import LEGACY_SCHEMA_VERSIONS, SCHEMA_VERSION
from gummybear.paths import repo_relative_path


REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "generator_version",
    "sequence_id",
    "created_utc",
    "forward_model_tier",
    "representation",
    "roles",
    "phantom",
    "workbook",
    "setups",
    "caches",
    "generation",
    "frames",
    "validation",
)
# Manifest keys required for generated-sequence artifact validation.
# Legacy top-level ``split`` / ``seed`` are optional and tolerated when present.

REQUIRED_ROLES = ("clean", "particle", "observed")

REQUIRED_SETUPS = ("optical", "particle", "diffusion", "camera", "corruption")

REQUIRED_FRAME_FIELDS = (
    "frame_index",
    "angle_deg",
    "axis",
    "camera_kind",
    "camera_position",
    "look_at",
    "up",
    "resolution",
    "fov_deg",
    "filenames",
)

EXPECTED_REPRESENTATION = {
    "image_format": "jpg",
    "jpeg_quality": 95,
    "image_domain": "camera_intensity",
    "composition_domain": "linear_camera_intensity_before_jpeg",
    "composition_policy": "pre_jpeg_numeric_arrays",
    "anomaly_definition": "particle_minus_clean",
    "observed_definition": "particle_no_corruption",
}

EXPECTED_PIXEL_ORIENTATION = {
    "camera_up": "image_top",
    "transform_from_camera_sample_grid": "flip_axis_0",
}

FRAME_TOKEN = re.compile(
    r"_frame_(?P<index>\d+)_angle_(?P<angle>[+-]?\d+(?:\.\d+)?)"
    r"\.(?:jpg|jpeg|png|raw\.tif)$",
    re.IGNORECASE,
)

FORBIDDEN_SCHEMA_ALIASES = ("schemaVersion", "manifest_version", "schema-version")

FORBIDDEN_RUNTIME_KEYS = {
    "fem_space",
    "factorized_operator",
    "netgen_object",
    "ngsolve_object",
    "operator_cache_id",
    "operator_cache_key",
    "operator_payload",
    "serialized_operator",
    "solver_handle",
}


@dataclass(frozen=True)
class SequenceValidationResult:
    """Structured pass/fail report for one generated sequence directory.

    **Pass:** ``ok`` is True (``errors`` empty). Warnings document checks that
    were skipped or approximate (for example post-display-JPEG identity only).

    Attributes:
        sequence_id: Manifest sequence id when readable.
        sequence_dir: Absolute path validated.
        ok: True when no errors were recorded.
        errors: Hard contract violations.
        warnings: Non-fatal notes (pre-JPEG float sidecar identity not checked, etc.).
        frame_count: Number of frames listed in the manifest.
        role_names: Role directories present (includes ``anomaly`` when claimed).
        checked_files: Manifest-relative paths opened during validation.
        manifest_schema_version: Parsed ``schema_version`` when present.
        composition_policy: ``representation.composition_policy`` when present.
        post_jpeg_identity_checked: True when observed≈particle display JPEG preview check ran.
        pre_jpeg_identity_checked: Always False when pre-JPEG float ``.raw.tif`` arrays are absent.
    """

    sequence_id: str | None
    sequence_dir: str
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    frame_count: int
    role_names: tuple[str, ...]
    checked_files: tuple[str, ...]
    manifest_schema_version: str | None
    composition_policy: str | None
    post_jpeg_identity_checked: bool
    pre_jpeg_identity_checked: bool

    def summary(self) -> dict[str, Any]:
        """Return a compact dict for notebook tables and logging.

        Paths are rewritten with :func:`gummybear.paths.repo_relative_path`.

        Returns:
            dict: JSON file-serializable summary with errors/warnings as lists.
        """
        return {
            "sequence_id": self.sequence_id,
            "sequence_dir": repo_relative_path(self.sequence_dir),
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "frame_count": self.frame_count,
            "role_names": list(self.role_names),
            "checked_file_count": len(self.checked_files),
            "manifest_schema_version": self.manifest_schema_version,
            "composition_policy": self.composition_policy,
            "post_jpeg_identity_checked": self.post_jpeg_identity_checked,
            "pre_jpeg_identity_checked": self.pre_jpeg_identity_checked,
        }


@dataclass
class _ValidationState:
    errors: list[str]
    warnings: list[str]
    checked_files: list[str]
    post_jpeg_checks: int = 0
    post_jpeg_checks_expected: int = 0

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _portable_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    has_windows_drive = bool(re.match(r"^[A-Za-z]:/", normalized))
    return not path.is_absolute() and not has_windows_drive and ".." not in path.parts


def _validate_sha256(value: Any, label: str, state: _ValidationState) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        state.error(f"{label} must be a 64-character SHA256 hex digest.")


def _validate_vector(
    value: Any,
    label: str,
    state: _ValidationState,
) -> None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not all(_is_finite_number(item) for item in value)
    ):
        state.error(f"{label} must contain three finite numbers.")


def _walk_forbidden_runtime_keys(
    value: Any,
    state: _ValidationState,
    *,
    location: str = "manifest",
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if str(key).lower() in FORBIDDEN_RUNTIME_KEYS:
                state.error(
                    f"{child_location} references forbidden serialized FEM/runtime state."
                )
            _walk_forbidden_runtime_keys(child, state, location=child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden_runtime_keys(
                child,
                state,
                location=f"{location}[{index}]",
            )


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    sequence_dir: Path,
    state: _ValidationState,
) -> None:
    for alias in FORBIDDEN_SCHEMA_ALIASES:
        if alias in manifest:
            state.error(
                f"Forbidden schema-version field {alias!r}; use 'schema_version'."
            )
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in manifest:
            state.error(f"Manifest missing required top-level field {field!r}.")

    schema_version = manifest.get("schema_version")
    if (
        schema_version != SCHEMA_VERSION
        and schema_version not in LEGACY_SCHEMA_VERSIONS
    ):
        state.error(
            "Unsupported schema_version "
            f"{schema_version!r}; expected {SCHEMA_VERSION!r} or a supported "
            f"legacy version {sorted(LEGACY_SCHEMA_VERSIONS)!r}."
        )
    if schema_version == SCHEMA_VERSION:
        _validate_sha256(
            manifest.get("resolved_job_hash"),
            "resolved_job_hash",
            state,
        )
        if not _is_mapping(manifest.get("resolved_job")):
            state.error("resolved_job must be an object for the current schema.")
    if not manifest.get("generator_version"):
        state.error("generator_version must be non-empty.")
    sequence_id = manifest.get("sequence_id")
    if not isinstance(sequence_id, str) or not sequence_id:
        state.error("sequence_id must be a non-empty string.")
    elif sequence_id != sequence_dir.name:
        state.error(
            f"sequence_id={sequence_id!r} does not match directory "
            f"name={sequence_dir.name!r}."
        )

    created_utc = manifest.get("created_utc")
    if not isinstance(created_utc, str):
        state.error("created_utc must be an ISO-8601 string.")
    else:
        try:
            datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
        except ValueError:
            state.error("created_utc must be a valid ISO-8601 timestamp.")
    if "seed" in manifest and not isinstance(manifest.get("seed"), int):
        state.error("legacy seed field must be an integer when present.")
    if "split" in manifest:
        split_value = manifest.get("split")
        if not isinstance(split_value, str) or not split_value:
            state.error("legacy split field must be a non-empty string when present.")

    phantom = manifest.get("phantom")
    if not _is_mapping(phantom):
        state.error("phantom must be an object.")
    else:
        stl_path = phantom.get("stl_path")
        if not _portable_relative_path(stl_path):
            state.error("phantom.stl_path must be a portable relative path.")
        _validate_sha256(phantom.get("stl_sha256"), "phantom.stl_sha256", state)

    workbook = manifest.get("workbook")
    if not _is_mapping(workbook):
        state.error("workbook must be an object.")
    else:
        workbook_path = workbook.get("workbook_path", workbook.get("path"))
        if not _portable_relative_path(workbook_path):
            state.error("workbook path must be a portable relative path.")
        _validate_sha256(workbook.get("sha256"), "workbook.sha256", state)

    setups = manifest.get("setups")
    if not _is_mapping(setups):
        state.error("setups must be an object.")
    else:
        for setup_name in REQUIRED_SETUPS:
            setup = setups.get(setup_name)
            label = f"setups.{setup_name}"
            if not _is_mapping(setup):
                state.error(f"{label} must be an object.")
                continue
            if not setup.get("workbook_name"):
                state.error(f"{label}.workbook_name is required.")
            else:
                workbook_name = str(setup["workbook_name"])
                normalized_name = workbook_name.replace("\\", "/")
                if (
                    not _portable_relative_path(workbook_name)
                    or PurePosixPath(normalized_name).name != normalized_name
                ):
                    state.error(f"{label}.workbook_name must be a portable filename.")
            if not (setup.get("workbook_sheet") or setup.get("source_sheet")):
                state.error(
                    f"{label} requires workbook_sheet or source_sheet provenance."
                )
            row = setup.get("source_excel_row")
            if not isinstance(row, int) or row < 2:
                state.error(f"{label}.source_excel_row must be an integer >= 2.")
        diffusion = setups.get("diffusion")
        if _is_mapping(diffusion) and not _is_mapping(diffusion.get("effective")):
            state.error("setups.diffusion.effective provenance must be an object.")

    caches = manifest.get("caches")
    if not _is_mapping(caches):
        state.error("caches must be an object.")
    else:
        for key in ("clean_optical_cache_id", "particle_source_cache_id"):
            if not isinstance(caches.get(key), str) or not caches.get(key):
                state.error(f"caches.{key} must be a non-empty string.")
        if caches.get("diffusion_operator_cache") is not None:
            state.error(
                "caches.diffusion_operator_cache must be null or absent; "
                "M6 must not persist FEM/operator state."
            )
        persistent = caches.get("persistent_cache_used")
        if persistent is not None and not isinstance(persistent, bool):
            state.error("caches.persistent_cache_used must be boolean when present.")
        events = caches.get("events")
        if persistent is True:
            if not _is_mapping(events):
                state.error(
                    "caches.events must be an object when persistent caches are used."
                )
            else:
                for name, id_key in (
                    ("clean_optical", "clean_optical_cache_id"),
                    ("particle_source", "particle_source_cache_id"),
                ):
                    event = events.get(name)
                    if not _is_mapping(event):
                        state.error(f"caches.events.{name} must be an object.")
                        continue
                    if event.get("cache_id") != caches.get(id_key):
                        state.error(
                            f"caches.events.{name}.cache_id must match caches.{id_key}."
                        )
                    if event.get("status") not in {"hit", "miss"}:
                        state.error(
                            f"caches.events.{name}.status must be 'hit' or 'miss'."
                        )
                    if not isinstance(event.get("reason"), str) or not event.get(
                        "reason"
                    ):
                        state.error(
                            f"caches.events.{name}.reason must be a non-empty string."
                        )

    generation = manifest.get("generation")
    if not _is_mapping(generation):
        state.error("generation must be an object.")
    else:
        timings = generation.get("stage_seconds", {})
        if not _is_mapping(timings):
            state.error("generation.stage_seconds must be an object.")
        else:
            for stage, seconds in timings.items():
                if not _is_finite_number(seconds) or float(seconds) < 0.0:
                    state.error(
                        f"generation.stage_seconds.{stage} must be non-negative."
                    )
        diagnostics = generation.get("diagnostics", {})
        if diagnostics is not None and not _is_mapping(diagnostics):
            state.error("generation.diagnostics must be an object when present.")
        elif _is_mapping(diagnostics):
            for name in (
                "n_source_rays",
                "n_refracted_rays",
                "n_source_segments",
                "n_affected_paths",
                "clean_solve_residual",
                "particle_solve_residual",
            ):
                if name in diagnostics and (
                    not _is_finite_number(diagnostics[name])
                    or float(diagnostics[name]) < 0.0
                ):
                    state.error(
                        f"generation.diagnostics.{name} must be finite and "
                        "non-negative."
                    )
            if (
                "source_assignment" in diagnostics
                and diagnostics["source_assignment"] != "attenuated_chord"
            ):
                state.error(
                    "generation.diagnostics.source_assignment must be "
                    "'attenuated_chord'."
                )

    _walk_forbidden_runtime_keys(manifest, state)


def _validate_representation(
    representation: Any,
    state: _ValidationState,
) -> None:
    if not _is_mapping(representation):
        state.error("representation must be an object.")
        return
    for field, expected in EXPECTED_REPRESENTATION.items():
        actual = representation.get(field)
        if actual != expected:
            state.error(f"representation.{field}={actual!r}; expected {expected!r}.")
    orientation = representation.get("pixel_orientation")
    if orientation != EXPECTED_PIXEL_ORIENTATION:
        state.error(
            f"representation.pixel_orientation must be {EXPECTED_PIXEL_ORIENTATION!r}."
        )
    anomaly_preview = representation.get("anomaly_preview")
    if not _is_mapping(anomaly_preview):
        state.error("representation.anomaly_preview must be an object.")
    else:
        if anomaly_preview.get("authoritative") is not False:
            state.error("representation.anomaly_preview.authoritative must be false.")
        if anomaly_preview.get("format") not in {None, "png"}:
            state.error("Anomaly preview format must be PNG or null.")
        if anomaly_preview.get("mapping") not in {
            None,
            "signed_per_frame_zero_centered",
        }:
            state.error("Unexpected anomaly preview mapping.")


def _role_directory(
    role: str,
    roles: Mapping[str, Any],
) -> str | None:
    if role == "anomaly":
        value = roles.get("anomaly", roles.get("anomaly_preview"))
    else:
        value = roles.get(role)
    return value if isinstance(value, str) else None


def _load_role_image(
    sequence_dir: Path,
    relative_name: str,
    *,
    role: str,
    frame_index: int,
    state: _ValidationState,
) -> tuple[np.ndarray, tuple[int, int], str] | None:
    path = sequence_dir / relative_name
    if not path.is_file():
        state.error(
            f"Frame {frame_index} role {role!r} file does not exist: {relative_name!r}."
        )
        return None
    if path.stat().st_size <= 0:
        state.error(
            f"Frame {frame_index} role {role!r} file is empty: {relative_name!r}."
        )
        return None
    state.checked_files.append(relative_name)
    try:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            size = image.size
            pixels = np.asarray(image)
    except (OSError, UnidentifiedImageError) as exc:
        state.error(
            f"Frame {frame_index} role {role!r} is not a readable image: "
            f"{relative_name!r} ({exc})."
        )
        return None
    if mode not in {"L", "RGB", "RGBA"}:
        state.error(f"Frame {frame_index} role {role!r} has unsupported mode {mode!r}.")
    return pixels, size, mode


def _validate_frames_and_roles(
    manifest: Mapping[str, Any],
    sequence_dir: Path,
    state: _ValidationState,
    *,
    post_jpeg_mean_tolerance: float,
    post_jpeg_max_tolerance: float,
) -> tuple[int, tuple[str, ...]]:
    roles = manifest.get("roles")
    if not _is_mapping(roles):
        state.error("roles must be an object.")
        return 0, ()

    role_names = tuple(
        role
        for role in (*REQUIRED_ROLES, "anomaly")
        if _role_directory(role, roles) is not None
    )
    for role in REQUIRED_ROLES:
        directory = _role_directory(role, roles)
        if directory is None:
            state.error(f"roles.{role} is required.")
            continue
        if not _portable_relative_path(directory):
            state.error(f"roles.{role} must be a portable relative path.")
            continue
        if not (sequence_dir / directory).is_dir():
            state.error(f"Required role directory is missing: {directory!r}.")

    anomaly_claimed = _role_directory("anomaly", roles) is not None
    if anomaly_claimed:
        anomaly_directory = _role_directory("anomaly", roles)
        if anomaly_directory is not None and (
            not _portable_relative_path(anomaly_directory)
            or not (sequence_dir / anomaly_directory).is_dir()
        ):
            state.error(
                f"Claimed anomaly preview directory is missing or non-portable: "
                f"{anomaly_directory!r}."
            )

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        state.error("frames must be a non-empty list.")
        return 0, role_names

    frame_indices = [
        frame.get("frame_index") if _is_mapping(frame) else None for frame in frames
    ]
    if any(not isinstance(index, int) for index in frame_indices):
        state.error("Every frame_index must be an integer.")
    else:
        if len(set(frame_indices)) != len(frame_indices):
            state.error("frame_index values must be unique.")
        if frame_indices != list(range(len(frames))):
            state.error(
                "Frame list must be ordered with sequential frame_index values "
                "starting at 0."
            )

    filenames_by_role: dict[str, list[str]] = {
        role: [] for role in (*REQUIRED_ROLES, "anomaly")
    }
    representation = manifest.get("representation")
    observed_definition = (
        representation.get("observed_definition")
        if _is_mapping(representation)
        else None
    )
    for position, frame in enumerate(frames):
        if not _is_mapping(frame):
            state.error(f"frames[{position}] must be an object.")
            continue
        for field in REQUIRED_FRAME_FIELDS:
            if field not in frame:
                state.error(f"frames[{position}] missing required field {field!r}.")
        frame_index = frame.get("frame_index")
        display_index = frame_index if isinstance(frame_index, int) else position
        angle_deg = frame.get("angle_deg")
        if not _is_finite_number(angle_deg):
            state.error(f"Frame {display_index} angle_deg must be finite.")
        for vector_name in ("axis", "camera_position", "look_at", "up"):
            _validate_vector(
                frame.get(vector_name),
                f"Frame {display_index} {vector_name}",
                state,
            )
        if not _is_finite_number(frame.get("fov_deg")):
            state.error(f"Frame {display_index} fov_deg must be finite.")
        if not isinstance(frame.get("camera_kind"), str) or not frame.get(
            "camera_kind"
        ):
            state.error(f"Frame {display_index} camera_kind must be non-empty.")

        resolution = frame.get("resolution")
        if (
            not isinstance(resolution, (list, tuple))
            or len(resolution) != 2
            or not all(isinstance(item, int) and item > 0 for item in resolution)
        ):
            state.error(
                f"Frame {display_index} resolution must contain two positive integers."
            )
            expected_size = None
        else:
            expected_size = (int(resolution[1]), int(resolution[0]))
            if tuple(resolution) != (128, 128):
                state.warning(
                    f"Frame {display_index} resolution is {tuple(resolution)}, "
                    "not the Phase 2 smoke resolution (128, 128)."
                )

        filenames = frame.get("filenames")
        if not _is_mapping(filenames):
            state.error(f"Frame {display_index} filenames must be an object.")
            continue
        expected_roles = list(REQUIRED_ROLES)
        if anomaly_claimed:
            expected_roles.append("anomaly")
        loaded: dict[str, tuple[np.ndarray, tuple[int, int], str]] = {}
        angle_tokens: set[str] = set()
        for role in expected_roles:
            relative_name = filenames.get(role)
            if not isinstance(relative_name, str):
                state.error(
                    f"Frame {display_index} missing filename for role {role!r}."
                )
                continue
            filenames_by_role[role].append(relative_name)
            if not _portable_relative_path(relative_name):
                state.error(
                    f"Frame {display_index} role {role!r} filename must be "
                    "sequence-relative."
                )
                continue
            expected_suffix = ".png" if role == "anomaly" else ".jpg"
            if PurePosixPath(relative_name).suffix.lower() != expected_suffix:
                state.error(
                    f"Frame {display_index} role {role!r} must use {expected_suffix}."
                )
            expected_directory = _role_directory(role, roles)
            if (
                expected_directory is not None
                and PurePosixPath(relative_name).parts[0] != expected_directory
            ):
                state.error(
                    f"Frame {display_index} role {role!r} filename is outside "
                    f"declared role directory {expected_directory!r}."
                )
            token = FRAME_TOKEN.search(relative_name)
            if token is None:
                state.error(
                    f"Frame {display_index} role {role!r} filename lacks the "
                    "required frame/angle tokens."
                )
            else:
                token_index = int(token.group("index"))
                if isinstance(frame_index, int) and token_index != frame_index:
                    state.error(
                        f"Frame {display_index} role {role!r} filename encodes "
                        f"frame index {token_index}."
                    )
                if len(token.group("index")) < 4:
                    state.error(
                        f"Frame {display_index} role {role!r} frame token must "
                        "be zero-padded to at least four digits."
                    )
                angle_tokens.add(token.group("angle"))
                if _is_finite_number(angle_deg):
                    try:
                        filename_angle = float(token.group("angle"))
                    except ValueError:
                        state.error(
                            f"Frame {display_index} role {role!r} angle token "
                            "is not numeric."
                        )
                    else:
                        if not math.isclose(
                            filename_angle,
                            float(angle_deg),
                            abs_tol=0.011,
                        ):
                            state.error(
                                f"Frame {display_index} role {role!r} angle "
                                "token does not match angle_deg."
                            )
            image = _load_role_image(
                sequence_dir,
                relative_name,
                role=role,
                frame_index=display_index,
                state=state,
            )
            if image is not None:
                loaded[role] = image
                if expected_size is not None and image[1] != expected_size:
                    state.error(
                        f"Frame {display_index} role {role!r} image size "
                        f"{image[1]} does not match manifest resolution "
                        f"{expected_size}."
                    )

        for role in REQUIRED_ROLES:
            relative_name = filenames.get(role)
            if not isinstance(relative_name, str):
                continue
            if not relative_name.lower().endswith(".jpg"):
                continue
            raw_relative = relative_name[: -len(".jpg")] + ".raw.tif"
            raw_path = sequence_dir / raw_relative
            if not raw_path.is_file():
                state.error(
                    f"Frame {display_index} role {role!r} missing float sidecar "
                    f"{raw_relative!r}."
                )
                continue
            state.checked_files.append(raw_relative)
            try:
                with Image.open(raw_path) as raw_image:
                    raw_image.load()
                    if raw_image.mode != "F":
                        state.error(
                            f"Frame {display_index} role {role!r} raw sidecar "
                            f"must be float mode F, got {raw_image.mode!r}."
                        )
                    if expected_size is not None and raw_image.size != expected_size:
                        state.error(
                            f"Frame {display_index} role {role!r} raw sidecar "
                            f"size {raw_image.size} does not match manifest "
                            f"resolution {expected_size}."
                        )
            except (OSError, UnidentifiedImageError) as exc:
                state.error(
                    f"Frame {display_index} role {role!r} raw sidecar is not "
                    f"readable: {raw_relative!r} ({exc})."
                )

        if len(angle_tokens) > 1:
            state.error(
                f"Frame {display_index} role filenames encode different angles."
            )
        loaded_sizes = {value[1] for value in loaded.values()}
        if len(loaded_sizes) > 1:
            state.error(f"Frame {display_index} role image dimensions do not align.")

        if observed_definition == "particle_no_corruption":
            state.post_jpeg_checks_expected += 1
            if "particle" in loaded and "observed" in loaded:
                particle = loaded["particle"][0].astype(float)
                observed = loaded["observed"][0].astype(float)
                if particle.shape == observed.shape:
                    difference = np.abs(observed - particle)
                    if (
                        float(np.mean(difference)) > post_jpeg_mean_tolerance
                        or float(np.max(difference)) > post_jpeg_max_tolerance
                    ):
                        state.error(
                            f"Frame {display_index} observed/particle post-JPEG "
                            "difference exceeds compression-aware tolerance."
                        )
                    state.post_jpeg_checks += 1

    for role, names in filenames_by_role.items():
        if names and names != sorted(names):
            state.error(f"Role {role!r} filenames do not sort in acquisition order.")
    return len(frames), role_names


def validate_generated_sequence(
    sequence_dir: str | Path,
    *,
    post_jpeg_mean_tolerance: float = 2.0,
    post_jpeg_max_tolerance: float = 8.0,
) -> SequenceValidationResult:
    """Validate one generated multi-view sequence directory (artifact-only).

    Checks manifest schema, portable paths, role filenames, float ``.raw.tif`` sidecars,
    display JPEG preview readability, and approximate observed/particle identity on decoded JPGs.
    Does **not** invoke forward-model physics or open pre-JPEG float sidecars
    (those are reported as not checked).

    **Pass:** ``errors`` is empty; ``ok`` is True. Post-display-JPEG observed/particle
    agreement uses compression-aware tolerances (mean ≤ ``post_jpeg_mean_tolerance``,
    max ≤ ``post_jpeg_max_tolerance`` by default).

    Args:
        sequence_dir: Path to ``<sequence_id>/`` containing ``manifest.json`` JSON file.
        post_jpeg_mean_tolerance: Mean absolute display JPEG preview difference tolerance.
        post_jpeg_max_tolerance: Max absolute display JPEG preview difference tolerance.

    Returns:
        SequenceValidationResult: Structured errors, warnings, and metadata.
    """
    directory = Path(sequence_dir)
    state = _ValidationState(errors=[], warnings=[], checked_files=[])
    manifest_path = directory / "manifest.json"
    if not directory.is_dir():
        state.error(f"Sequence directory does not exist: {directory}.")
        return SequenceValidationResult(
            sequence_id=None,
            sequence_dir=str(directory),
            ok=False,
            errors=tuple(state.errors),
            warnings=(),
            frame_count=0,
            role_names=(),
            checked_files=(),
            manifest_schema_version=None,
            composition_policy=None,
            post_jpeg_identity_checked=False,
            pre_jpeg_identity_checked=False,
        )
    if not manifest_path.is_file():
        state.error("manifest.json is missing.")
        return SequenceValidationResult(
            sequence_id=directory.name,
            sequence_dir=str(directory),
            ok=False,
            errors=tuple(state.errors),
            warnings=(),
            frame_count=0,
            role_names=(),
            checked_files=(),
            manifest_schema_version=None,
            composition_policy=None,
            post_jpeg_identity_checked=False,
            pre_jpeg_identity_checked=False,
        )

    state.checked_files.append("manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        state.error(f"manifest.json is not valid readable JSON: {exc}.")
        return SequenceValidationResult(
            sequence_id=directory.name,
            sequence_dir=str(directory),
            ok=False,
            errors=tuple(state.errors),
            warnings=(),
            frame_count=0,
            role_names=(),
            checked_files=tuple(state.checked_files),
            manifest_schema_version=None,
            composition_policy=None,
            post_jpeg_identity_checked=False,
            pre_jpeg_identity_checked=False,
        )
    if not _is_mapping(manifest):
        state.error("manifest.json root must be an object.")
        manifest = {}

    _validate_manifest_contract(manifest, directory, state)
    _validate_representation(manifest.get("representation"), state)
    frame_count, role_names = _validate_frames_and_roles(
        manifest,
        directory,
        state,
        post_jpeg_mean_tolerance=post_jpeg_mean_tolerance,
        post_jpeg_max_tolerance=post_jpeg_max_tolerance,
    )

    composition_policy = (
        manifest.get("representation", {}).get("composition_policy")
        if _is_mapping(manifest.get("representation"))
        else None
    )
    pre_jpeg_checked = False
    state.warning(
        "Pre-JPEG anomaly/composition identity not checked: current M6.2 "
        "artifacts do not persist numeric pre-compression arrays."
    )
    post_jpeg_checked = (
        state.post_jpeg_checks_expected > 0
        and state.post_jpeg_checks == state.post_jpeg_checks_expected
    )
    if post_jpeg_checked:
        state.warning(
            "Observed/particle identity was checked approximately on decoded "
            "JPEG files; this is not an exact pre-JPEG residual check."
        )

    return SequenceValidationResult(
        sequence_id=(
            manifest.get("sequence_id")
            if isinstance(manifest.get("sequence_id"), str)
            else None
        ),
        sequence_dir=str(directory),
        ok=not state.errors,
        errors=tuple(state.errors),
        warnings=tuple(state.warnings),
        frame_count=frame_count,
        role_names=role_names,
        checked_files=tuple(state.checked_files),
        manifest_schema_version=(
            manifest.get("schema_version")
            if isinstance(manifest.get("schema_version"), str)
            else None
        ),
        composition_policy=(
            composition_policy if isinstance(composition_policy, str) else None
        ),
        post_jpeg_identity_checked=post_jpeg_checked,
        pre_jpeg_identity_checked=pre_jpeg_checked,
    )
