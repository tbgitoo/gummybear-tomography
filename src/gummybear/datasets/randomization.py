"""Offline workbook particle-centre and train/val/test split randomization.

Runtime generation stays ``placement_mode=fixed``. These helpers author Excel
control fields (particle centres; sequence ``split`` / ``seed``) without running
optical physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gummybear.datasets.generation_workbook import (
    REQUIRED_SHEETS,
    WorkbookValidationError,
    normalize_enabled,
    write_generation_workbook_frames,
)
from gummybear.geometry import load_stl
from gummybear.particles import sample_random_centers_in_mesh

# Default particle-level train / validation / test fractions (70% / 15% / 15%).
DEFAULT_SPLIT_FRACTIONS: dict[str, float] = {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15,
}

_SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class ParticleRandomizationResult:
    """Outcome of rewriting particle centres in a generation workbook.

    Attributes:
        workbook_path: Resolved workbook path that was updated.
        seed: RNG seed used for the placement stream.
        stl_path: Relative mesh path taken from the sequences sheet.
        mesh_path: Absolute mesh path resolved under ``root``.
        n_particles: Number of particle rows that received new centres.
        centers: Float64 array of shape ``(n_particles, 3)`` in sheet order.
    """

    workbook_path: Path
    seed: int
    stl_path: str
    mesh_path: Path
    n_particles: int
    centers: np.ndarray


@dataclass(frozen=True)
class SplitRandomizationResult:
    """Outcome of rewriting sequence ``split`` values stratified by particle.

    Attributes:
        workbook_path: Resolved workbook path that was updated.
        seed: RNG seed used for the particle→split shuffle (also written to
            ``sequences.seed``).
        n_sequences: Number of sequence rows rewritten.
        n_particles: Number of unique ``particle_setup_id`` values assigned.
        split_fractions: Normalized train / validation / test fractions used.
        particle_splits: Mapping particle_setup_id → split label.
        split_counts: Counts of rewritten sequence rows per split label.
    """

    workbook_path: Path
    seed: int
    n_sequences: int
    n_particles: int
    split_fractions: dict[str, float]
    particle_splits: dict[str, str]
    split_counts: dict[str, int]


def _require_pandas_openpyxl():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise WorkbookValidationError(
            "pandas is required to randomize generation workbooks."
        ) from exc
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise WorkbookValidationError(
            "openpyxl is required to randomize generation workbooks. "
            "From the repository root, install it with:\n"
            "  pip install openpyxl -c requirements.txt"
        ) from exc
    return pd


def _unique_stl_path(sequences) -> str:
    """Return the single ``stl_path`` used by the sequences sheet."""
    if sequences.empty:
        raise WorkbookValidationError(
            "sequences sheet is empty; cannot resolve stl_path for placement."
        )
    if "stl_path" not in sequences.columns:
        raise WorkbookValidationError(
            "sequences sheet is missing required column 'stl_path'."
        )

    rows = sequences
    if "enabled" in sequences.columns:
        enabled_mask = sequences["enabled"].map(
            lambda value: normalize_enabled(value, default=True)
        )
        if bool(enabled_mask.any()):
            rows = sequences.loc[enabled_mask]

    paths: list[str] = []
    for value in rows["stl_path"].tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise WorkbookValidationError(
                "sequences.stl_path contains an empty value."
            )
        text = str(value).strip()
        if not text:
            raise WorkbookValidationError(
                "sequences.stl_path contains an empty value."
            )
        paths.append(text)

    unique = sorted(set(paths))
    if len(unique) != 1:
        raise WorkbookValidationError(
            "randomize_workbook_particle_centers requires a single unique "
            f"stl_path among selected sequences rows (got {unique!r})."
        )
    return unique[0]


def _resolve_mesh_path(root: Path | str, stl_path: str) -> Path:
    root_path = Path(root)
    mesh_path = (root_path / stl_path).resolve()
    if not mesh_path.is_file():
        raise WorkbookValidationError(
            f"Mesh STL not found at {mesh_path} "
            f"(root={root_path.resolve()}, stl_path={stl_path!r})."
        )
    return mesh_path


def _normalize_split_fractions(
    split_fractions: Mapping[str, float] | None,
) -> dict[str, float]:
    raw = dict(DEFAULT_SPLIT_FRACTIONS if split_fractions is None else split_fractions)
    missing = [name for name in _SPLIT_ORDER if name not in raw]
    if missing:
        raise WorkbookValidationError(
            "split_fractions must include "
            f"{list(_SPLIT_ORDER)} (missing {missing!r})."
        )
    values = {name: float(raw[name]) for name in _SPLIT_ORDER}
    if any(v < 0.0 for v in values.values()):
        raise WorkbookValidationError("split_fractions must be non-negative.")
    total = float(sum(values.values()))
    if total <= 0.0:
        raise WorkbookValidationError("split_fractions must sum to a positive value.")
    return {name: values[name] / total for name in _SPLIT_ORDER}


def _largest_remainder_counts(n: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Allocate ``n`` integer seats by largest-remainder from unit fractions."""
    if n < 0:
        raise WorkbookValidationError("n must be non-negative")
    if n == 0:
        return {name: 0 for name in _SPLIT_ORDER}

    exact = {name: fractions[name] * n for name in _SPLIT_ORDER}
    floors = {name: int(np.floor(exact[name])) for name in _SPLIT_ORDER}
    assigned = int(sum(floors.values()))
    remainders = sorted(
        _SPLIT_ORDER,
        key=lambda name: (exact[name] - floors[name], name),
        reverse=True,
    )
    counts = dict(floors)
    for name in remainders[: max(0, n - assigned)]:
        counts[name] += 1
    if sum(counts.values()) != n:
        raise RuntimeError("internal split allocation error")
    return counts


def assign_particle_splits(
    particle_setup_ids: list[str] | tuple[str, ...],
    seed: int,
    *,
    split_fractions: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Assign each unique particle id to train / validation / test.

    Unique ids keep first-seen order, then a seeded shuffle decides membership.
    All sequences sharing a ``particle_setup_id`` should use the returned label.
    """
    fractions = _normalize_split_fractions(split_fractions)
    unique_ids = list(dict.fromkeys(str(pid) for pid in particle_setup_ids))
    n = len(unique_ids)
    counts = _largest_remainder_counts(n, fractions)
    rng = np.random.default_rng(int(seed))
    order = np.arange(n)
    rng.shuffle(order)
    shuffled = [unique_ids[int(i)] for i in order]

    assignment: dict[str, str] = {}
    start = 0
    for split_name in _SPLIT_ORDER:
        stop = start + counts[split_name]
        for particle_id in shuffled[start:stop]:
            assignment[particle_id] = split_name
        start = stop
    return assignment


def randomize_workbook_particle_centers(
    workbook_path: Path | str,
    seed: int,
    root: Path | str,
    *,
    only_enabled: bool = False,
    min_center_separation: float | None = None,
    radius: float | None = None,
    max_attempts: int | None = None,
) -> ParticleRandomizationResult:
    """Overwrite ``particles`` centres with a seeded mesh-volume sample.

    Reads the generation workbook at ``workbook_path``, resolves the mesh from
    the unique ``sequences.stl_path`` relative to ``root``, samples one centre
    per configured particle row from ``numpy.random.default_rng(seed)`` (via
    :func:`~gummybear.particles.sample_random_centers_in_mesh`), writes
    ``center_x`` / ``center_y`` / ``center_z`` and records ``seed`` on each
    updated particle row, then saves the workbook in place.

    The same ``seed`` reproduces the same centre sequence in sheet row order.

    Parameters
    ----------
    workbook_path
        Path to an M6/M8 generation Excel workbook.
    seed
        Placement RNG seed (start of the random stream).
    root
        Directory relative to which ``sequences.stl_path`` is resolved.
    only_enabled
        When True, only rows with ``enabled`` truthy are rewritten; disabled
        rows keep their previous centres and seed. Default False rewrites every
        particle sheet row.
    min_center_separation, radius, max_attempts
        Forwarded to :func:`~gummybear.particles.sample_random_centers_in_mesh`.
        Leave separation unset for independent single-particle catalogue rows
        that never co-occur in one scene.
    """
    pd = _require_pandas_openpyxl()
    path = Path(workbook_path).resolve()
    if not path.is_file():
        raise WorkbookValidationError(f"Workbook not found: {path}")

    try:
        frames: dict[str, Any] = pd.read_excel(
            path,
            sheet_name=list(REQUIRED_SHEETS),
            engine="openpyxl",
        )
    except ValueError as exc:
        raise WorkbookValidationError(
            f"Failed to read required sheets from {path}: {exc}"
        ) from exc

    stl_path = _unique_stl_path(frames["sequences"])
    mesh_path = _resolve_mesh_path(root, stl_path)
    mesh = load_stl(mesh_path)

    particles = frames["particles"].copy()
    if particles.empty:
        return ParticleRandomizationResult(
            workbook_path=path,
            seed=int(seed),
            stl_path=stl_path,
            mesh_path=mesh_path,
            n_particles=0,
            centers=np.zeros((0, 3), dtype=float),
        )

    for column in ("center_x", "center_y", "center_z", "seed"):
        if column not in particles.columns:
            raise WorkbookValidationError(
                f"particles sheet is missing required column {column!r}."
            )

    if only_enabled:
        if "enabled" not in particles.columns:
            raise WorkbookValidationError(
                "only_enabled=True requires an 'enabled' column on particles."
            )
        target_mask = particles["enabled"].map(
            lambda value: normalize_enabled(value, default=True)
        )
    else:
        target_mask = pd.Series(True, index=particles.index)

    target_indices = particles.index[target_mask].tolist()
    n = len(target_indices)
    centers = sample_random_centers_in_mesh(
        mesh,
        n,
        seed=int(seed),
        min_center_separation=min_center_separation,
        radius=radius,
        max_attempts=max_attempts,
    )

    for row_i, center in zip(target_indices, centers, strict=True):
        particles.loc[row_i, "center_x"] = float(center[0])
        particles.loc[row_i, "center_y"] = float(center[1])
        particles.loc[row_i, "center_z"] = float(center[2])
        particles.loc[row_i, "seed"] = int(seed)

    frames["particles"] = particles
    write_generation_workbook_frames(path, frames)

    return ParticleRandomizationResult(
        workbook_path=path,
        seed=int(seed),
        stl_path=stl_path,
        mesh_path=mesh_path,
        n_particles=n,
        centers=np.asarray(centers, dtype=float).reshape(n, 3),
    )


def randomize_workbook_sequence_splits(
    workbook_path: Path | str,
    seed: int,
    *,
    train_fraction: float = DEFAULT_SPLIT_FRACTIONS["train"],
    validation_fraction: float = DEFAULT_SPLIT_FRACTIONS["validation"],
    test_fraction: float = DEFAULT_SPLIT_FRACTIONS["test"],
    split_fractions: Mapping[str, float] | None = None,
    only_enabled: bool = False,
) -> SplitRandomizationResult:
    """Overwrite ``sequences.split`` stratified by ``particle_setup_id``.

    All sequence rows that share a ``particle_setup_id`` receive the same split
    label so the same particle under different optical/camera conditions never
    leaks across train / validation / test. The RNG ``seed`` is written to
    ``sequences.seed`` on every updated row.

    Default fractions are 70% / 15% / 15% (train / validation / test).
    Pass ``train_fraction`` / ``validation_fraction`` / ``test_fraction``, or a
    single ``split_fractions`` mapping, to override.

    The same ``seed`` and fractions reproduce the same particle→split mapping.
    """
    pd = _require_pandas_openpyxl()
    path = Path(workbook_path).resolve()
    if not path.is_file():
        raise WorkbookValidationError(f"Workbook not found: {path}")

    try:
        frames: dict[str, Any] = pd.read_excel(
            path,
            sheet_name=list(REQUIRED_SHEETS),
            engine="openpyxl",
        )
    except ValueError as exc:
        raise WorkbookValidationError(
            f"Failed to read required sheets from {path}: {exc}"
        ) from exc

    sequences = frames["sequences"].copy()
    fractions = _normalize_split_fractions(
        split_fractions
        if split_fractions is not None
        else {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        }
    )
    if sequences.empty:
        return SplitRandomizationResult(
            workbook_path=path,
            seed=int(seed),
            n_sequences=0,
            n_particles=0,
            split_fractions=fractions,
            particle_splits={},
            split_counts={name: 0 for name in _SPLIT_ORDER},
        )

    for column in ("split", "seed", "particle_setup_id"):
        if column not in sequences.columns:
            raise WorkbookValidationError(
                f"sequences sheet is missing required column {column!r}."
            )

    if only_enabled:
        if "enabled" not in sequences.columns:
            raise WorkbookValidationError(
                "only_enabled=True requires an 'enabled' column on sequences."
            )
        target_mask = sequences["enabled"].map(
            lambda value: normalize_enabled(value, default=True)
        )
    else:
        target_mask = pd.Series(True, index=sequences.index)

    target_indices = sequences.index[target_mask].tolist()
    particle_ids = [
        str(sequences.loc[row_i, "particle_setup_id"]) for row_i in target_indices
    ]
    particle_splits = assign_particle_splits(
        particle_ids,
        seed=int(seed),
        split_fractions=fractions,
    )

    split_counts = {name: 0 for name in _SPLIT_ORDER}
    for row_i in target_indices:
        particle_id = str(sequences.loc[row_i, "particle_setup_id"])
        split_name = particle_splits[particle_id]
        sequences.loc[row_i, "split"] = split_name
        sequences.loc[row_i, "seed"] = int(seed)
        split_counts[split_name] += 1

    frames["sequences"] = sequences
    write_generation_workbook_frames(path, frames)

    return SplitRandomizationResult(
        workbook_path=path,
        seed=int(seed),
        n_sequences=len(target_indices),
        n_particles=len(particle_splits),
        split_fractions=fractions,
        particle_splits=particle_splits,
        split_counts=split_counts,
    )
