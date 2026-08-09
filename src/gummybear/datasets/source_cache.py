"""Versioned persistence for clean, particle, and camera-visibility numeric caches.

Each cache entry is an ``.npz`` payload plus a completed ``.json`` sidecar.
Publication writes payload first and marks completion last for crash safety.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from gummybear import __version__


CLEAN_PAYLOAD_SCHEMA_VERSION = "m6-clean-payload-v1"
PARTICLE_PAYLOAD_SCHEMA_VERSION = "m6-particle-payload-v1"
CAMERA_VISIBILITY_PAYLOAD_SCHEMA_VERSION = "m6-camera-visibility-payload-v1"
PHI_SAMPLING_LOCALIZATION_PAYLOAD_SCHEMA_VERSION = (
    "m6-phi-sampling-localization-payload-v1"
)

CLEAN_REQUIRED_ARRAYS = (
    "source_origins",
    "source_directions",
    "source_weights",
    "source_sample_shape",
    "refracted_origins",
    "refracted_directions",
    "refracted_weights",
    "refracted_sample_shape",
    "refracted_parent_indices",
    "refracted_valid_mask",
    "refracted_hit_faces",
    "refracted_hit_points",
    "refracted_full_directions",
    "segment_starts",
    "segment_ends",
    "segment_intensities",
    "segment_ray_ids",
    "segment_ids",
    "segment_path_order",
    "S_clean",
    "E_scat_elem",
)
CLEAN_ALLOW_NONFINITE_ARRAYS = ("refracted_hit_points",)

PARTICLE_REQUIRED_ARRAYS = (
    "E_clean_elem",
    "delta_E_background_elem",
    "delta_E_particle_scat_elem",
    "delta_E_transport_elem",
    "E_particle_elem",
    "S_particle",
    "affected_path_ids",
    "affected_segment_indices",
    "particle_ray_weights",
)

CAMERA_VISIBILITY_REQUIRED_ARRAYS = (
    "valid_mask",
    "hit_depth",
    "hit_faces",
    "hit_points",
    "sample_shape",
)
CAMERA_VISIBILITY_ALLOW_NONFINITE_ARRAYS = ("hit_depth", "hit_points")

PHI_SAMPLING_LOCALIZATION_REQUIRED_ARRAYS = (
    "sample_mode",
    "tet_id",
    "barycentric",
    "sample_shape",
)


@dataclass(frozen=True)
class CacheEvent:
    """Structured cache probe or publish result for manifests and notebooks.

    Attributes:
        kind: Cache namespace (``clean_optical``, ``particle_source``, etc.).
        cache_id: Full SHA256 cache key digest.
        status: ``hit``, ``miss``, ``mixed``, or ``disabled``.
        reason: Machine-readable status detail.
        payload_path, sidecar_path: On-disk pair paths when known.
        load_seconds, write_seconds: Timing for profiling.
    """

    kind: str
    cache_id: str
    status: str
    reason: str
    payload_path: str | None
    sidecar_path: str | None
    load_seconds: float = 0.0
    write_seconds: float = 0.0

    @property
    def hit(self) -> bool:
        """True when a valid cache pair was loaded successfully.

        Equivalent to ``status == "hit"``; ``mixed`` and ``miss`` are False.
        """
        return self.status == "hit"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON file-friendly event record."""
        return {
            "kind": self.kind,
            "cache_id": self.cache_id,
            "status": self.status,
            "reason": self.reason,
            "payload_path": self.payload_path,
            "sidecar_path": self.sidecar_path,
            "load_seconds": float(self.load_seconds),
            "write_seconds": float(self.write_seconds),
        }


@dataclass(frozen=True)
class CacheLoadResult:
    """Result of attempting to load one cache pair.

    Attributes:
        event: Structured status (hit or miss with reason).
        arrays: Loaded numpy arrays when ``event.hit`` is True.
        payload_metadata: Sidecar metadata dict when load succeeds.
    """

    event: CacheEvent
    arrays: dict[str, np.ndarray] | None = None
    payload_metadata: dict[str, Any] | None = None


class SourceCacheStore:
    """Read and atomically publish versioned source-cache ``.npz``/``.json`` pairs.

    Args:
        root: Scenario ``_cache`` directory (e.g. ``data/generated/m6_5/_cache``).
    """

    def __init__(self, root: Path | str):
        """Bind a scenario cache directory for load and write operations.

        Args:
            root: Scenario ``_cache`` directory (e.g. ``data/generated/m6_5/_cache``).
                Subdirectories are created per cache ``kind`` on first write.
        """
        self.root = Path(root)

    def _paths(self, kind: str, cache_id: str) -> tuple[Path, Path]:
        directory = self.root / kind
        return directory / f"{cache_id}.npz", directory / f"{cache_id}.json"

    def load(
        self,
        *,
        kind: str,
        cache_id: str,
        key_payload: Mapping[str, Any],
        payload_schema_version: str,
        required_arrays: tuple[str, ...],
        mesh_identity: Mapping[str, Any] | None,
        force_recompute: bool = False,
        allow_nonfinite_arrays: tuple[str, ...] = (),
    ) -> CacheLoadResult:
        """Load a validated cache pair, returning a structured miss otherwise.

        Args:
            kind: Cache subdirectory name.
            cache_id: Expected digest; must match sidecar ``cache_id``.
            key_payload: Canonical key dict; must match sidecar ``key_payload``.
            payload_schema_version: Required sidecar schema version.
            required_arrays: Array names that must exist with matching dtype/shape.
            mesh_identity: Optional diffusion mesh identity for alignment checks.
            force_recompute: When True, behave as a miss regardless of files.
            allow_nonfinite_arrays: Array names permitted to contain non-finite values.

        Returns:
            CacheLoadResult with ``event.hit`` True only when validation passes.
        """
        payload_path, sidecar_path = self._paths(kind, cache_id)
        base = CacheEvent(
            kind=kind,
            cache_id=cache_id,
            status="miss",
            reason="miss_not_found",
            payload_path=str(payload_path),
            sidecar_path=str(sidecar_path),
        )
        if force_recompute:
            return CacheLoadResult(
                replace(base, reason="forced_recompute")
            )

        started = perf_counter()
        payload_exists = payload_path.is_file()
        sidecar_exists = sidecar_path.is_file()
        if not payload_exists and not sidecar_exists:
            return CacheLoadResult(
                replace(base, load_seconds=perf_counter() - started)
            )
        if not payload_exists or not sidecar_exists:
            return CacheLoadResult(
                replace(
                    base,
                    reason="miss_incomplete",
                    load_seconds=perf_counter() - started,
                )
            )

        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return CacheLoadResult(
                replace(
                    base,
                    reason="miss_payload_invalid",
                    load_seconds=perf_counter() - started,
                )
            )

        if not isinstance(sidecar, dict) or sidecar.get("completion_status") != "complete":
            return CacheLoadResult(
                replace(
                    base,
                    reason="miss_incomplete",
                    load_seconds=perf_counter() - started,
                )
            )
        if (
            sidecar.get("cache_kind") != kind
            or sidecar.get("cache_id") != cache_id
            or sidecar.get("key_payload") != dict(key_payload)
        ):
            return CacheLoadResult(
                replace(
                    base,
                    reason="miss_key_mismatch",
                    load_seconds=perf_counter() - started,
                )
            )
        if sidecar.get("payload_schema_version") != payload_schema_version:
            return CacheLoadResult(
                replace(
                    base,
                    reason="miss_schema_or_algorithm_mismatch",
                    load_seconds=perf_counter() - started,
                )
            )
        if (
            mesh_identity is not None
            and sidecar.get("diffusion_mesh_identity") != dict(mesh_identity)
        ):
            return CacheLoadResult(
                replace(
                    base,
                    reason="miss_mesh_alignment_mismatch",
                    load_seconds=perf_counter() - started,
                )
            )

        try:
            declared_specs = sidecar["arrays"]
            if not isinstance(declared_specs, dict):
                raise ValueError("arrays must be an object")
            with np.load(payload_path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
            for name in required_arrays:
                if name not in arrays or name not in declared_specs:
                    raise ValueError(f"missing required array {name}")
            for name, array in arrays.items():
                spec = declared_specs.get(name)
                if not isinstance(spec, dict):
                    raise ValueError(f"missing array specification for {name}")
                if list(array.shape) != spec.get("shape"):
                    raise ValueError(f"shape mismatch for {name}")
                if str(array.dtype) != spec.get("dtype"):
                    raise ValueError(f"dtype mismatch for {name}")
                if (
                    name not in allow_nonfinite_arrays
                    and array.dtype.kind in {"f", "c"}
                    and not np.all(np.isfinite(array))
                ):
                    raise ValueError(f"non-finite values in {name}")
            payload_metadata = sidecar.get("payload_metadata", {})
            if not isinstance(payload_metadata, dict):
                raise ValueError("payload_metadata must be an object")
        except (OSError, KeyError, ValueError):
            return CacheLoadResult(
                replace(
                    base,
                    reason="miss_payload_invalid",
                    load_seconds=perf_counter() - started,
                )
            )

        return CacheLoadResult(
            event=replace(
                base,
                status="hit",
                reason="hit",
                load_seconds=perf_counter() - started,
            ),
            arrays=arrays,
            payload_metadata=payload_metadata,
        )

    def write(
        self,
        *,
        kind: str,
        cache_id: str,
        key_payload: Mapping[str, Any],
        payload_schema_version: str,
        arrays: Mapping[str, np.ndarray],
        payload_metadata: Mapping[str, Any],
        mesh_identity: Mapping[str, Any],
        workbook_provenance: Mapping[str, Any],
        prior_event: CacheEvent,
    ) -> CacheEvent:
        """Publish payload first and the completed sidecar last.

        Args:
            kind, cache_id, key_payload, payload_schema_version: Cache identity.
            arrays: Named numpy arrays written to compressed ``.npz``.
            payload_metadata: Extra metadata stored in the sidecar.
            mesh_identity: Diffusion mesh identity recorded for alignment.
            workbook_provenance: Workbook coordinates for audit trails.
            prior_event: Miss event to upgrade with paths and write timing.

        Returns:
            Updated CacheEvent with ``payload_path``, ``sidecar_path``, and timing.
        """
        payload_path, sidecar_path = self._paths(kind, cache_id)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        temporary_payload = payload_path.with_name(f".{payload_path.name}.{token}.tmp.npz")
        temporary_sidecar = sidecar_path.with_name(
            f".{sidecar_path.name}.{token}.tmp"
        )
        normalized_arrays = {
            name: np.asarray(value) for name, value in arrays.items()
        }
        sidecar = {
            "cache_kind": kind,
            "cache_id": cache_id,
            "key_payload": dict(key_payload),
            "algorithm_version": key_payload.get("algorithm_version"),
            "payload_schema_version": payload_schema_version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "package_version": __version__,
            "completion_status": "complete",
            "diffusion_mesh_identity": dict(mesh_identity),
            "arrays": {
                name: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
                for name, array in sorted(normalized_arrays.items())
            },
            "payload_metadata": dict(payload_metadata),
            "workbook_provenance": dict(workbook_provenance),
        }
        if kind == "particle_source":
            sidecar["parent_clean_cache_id"] = key_payload.get(
                "clean_optical_cache_id"
            )

        started = perf_counter()
        try:
            np.savez_compressed(temporary_payload, **normalized_arrays)
            temporary_sidecar.write_text(
                json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_payload, payload_path)
            os.replace(temporary_sidecar, sidecar_path)
        finally:
            temporary_payload.unlink(missing_ok=True)
            temporary_sidecar.unlink(missing_ok=True)

        return replace(
            prior_event,
            payload_path=str(payload_path),
            sidecar_path=str(sidecar_path),
            write_seconds=perf_counter() - started,
        )
