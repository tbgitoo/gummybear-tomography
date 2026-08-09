"""Lazy task datasets over flat catalog rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from PIL import Image

from gummybear.datasets.role_images import role_image_relative_to_raw_tif
from tomography_ml.gummybear_data_catalog.catalog import CatalogRow, RoleRef

ImageRepresentation = Literal["raw_float", "jpeg_uint8"]
ImageNormalize = Literal[
    "none",
    "per_image_minmax",
    "per_image_zscore",
    "train_split_zscore",
]

IMAGE_REPRESENTATION_RAW_FLOAT: ImageRepresentation = "raw_float"
IMAGE_REPRESENTATION_JPEG_UINT8: ImageRepresentation = "jpeg_uint8"

# Default for new loaders: linear float32 ``.raw.tif`` sidecars.
DEFAULT_IMAGE_REPRESENTATION: ImageRepresentation = IMAGE_REPRESENTATION_RAW_FLOAT

# Historical validation path: uint8 JPG display copies.
M7_IMAGE_REPRESENTATION: ImageRepresentation = IMAGE_REPRESENTATION_JPEG_UINT8

IMAGE_NORMALIZE_NONE: ImageNormalize = "none"
IMAGE_NORMALIZE_PER_IMAGE_MINMAX: ImageNormalize = "per_image_minmax"
IMAGE_NORMALIZE_PER_IMAGE_ZSCORE: ImageNormalize = "per_image_zscore"
IMAGE_NORMALIZE_TRAIN_SPLIT_ZSCORE: ImageNormalize = "train_split_zscore"
DEFAULT_IMAGE_NORMALIZE: ImageNormalize = IMAGE_NORMALIZE_NONE

SUPPORTED_IMAGE_NORMALIZE: frozenset[str] = frozenset(
    {
        IMAGE_NORMALIZE_NONE,
        IMAGE_NORMALIZE_PER_IMAGE_MINMAX,
        IMAGE_NORMALIZE_PER_IMAGE_ZSCORE,
        IMAGE_NORMALIZE_TRAIN_SPLIT_ZSCORE,
    }
)

_ROLES_WITH_RAW_FLOAT = frozenset({"clean", "particle", "observed", "anomaly"})

DEFAULT_ANGLE_ATOL_DEG = 1e-6
DEFAULT_INTENSITY_STD_EPS = 1e-8


@dataclass(frozen=True)
class IntensityStats:
    """Train-split global intensity mean / std for ``train_split_zscore``."""

    mean: float
    std: float

    def to_dict(self) -> dict[str, float]:
        """Serialize train-split intensity statistics for configs or sidecars.

        Returns:
            Dict with ``mean`` and ``std`` keys (floats).
        """
        return {"mean": float(self.mean), "std": float(self.std)}


def apply_image_normalize(
    array: np.ndarray,
    image_normalize: ImageNormalize,
    *,
    intensity_stats: IntensityStats | None = None,
    std_eps: float = DEFAULT_INTENSITY_STD_EPS,
) -> np.ndarray:
    """Apply optional intensity normalisation to a ``(V, C, H, W)`` role array.

    Modes:

    - ``none`` — leave linear intensities unchanged.
    - ``per_image_minmax`` — each view independently to ``[0, 1]`` via
      ``(x - min) / (max - min)``. Constant views become zeros.
    - ``per_image_zscore`` — each view independently ``(x - mean) / std``.
      Constant views become zeros.
    - ``train_split_zscore`` — global ``(x - mean) / std`` using
      ``intensity_stats`` estimated on the training split only.
    """
    if image_normalize == IMAGE_NORMALIZE_NONE:
        return array
    if image_normalize not in SUPPORTED_IMAGE_NORMALIZE:
        raise ValueError(
            f"Unsupported image_normalize={image_normalize!r}; "
            f"expected one of {sorted(SUPPORTED_IMAGE_NORMALIZE)}."
        )
    if array.ndim != 4:
        raise ValueError(
            f"{image_normalize} expects role arrays with shape (V, C, H, W); "
            f"got shape {tuple(array.shape)}"
        )

    if image_normalize == IMAGE_NORMALIZE_TRAIN_SPLIT_ZSCORE:
        if intensity_stats is None:
            raise ValueError(
                "train_split_zscore requires intensity_stats "
                "(estimate on the train split with estimate_intensity_stats)."
            )
        mean = float(intensity_stats.mean)
        std = max(float(intensity_stats.std), float(std_eps))
        return ((np.asarray(array, dtype=np.float32) - mean) / std).astype(
            np.float32, copy=False
        )

    out = np.empty(array.shape, dtype=np.float32)
    for view_index in range(array.shape[0]):
        view = np.asarray(array[view_index], dtype=np.float32)
        if image_normalize == IMAGE_NORMALIZE_PER_IMAGE_MINMAX:
            lo = float(np.min(view))
            hi = float(np.max(view))
            if hi <= lo:
                out[view_index] = 0.0
            else:
                out[view_index] = (view - lo) / (hi - lo)
        elif image_normalize == IMAGE_NORMALIZE_PER_IMAGE_ZSCORE:
            mean = float(np.mean(view))
            std = float(np.std(view))
            if std <= float(std_eps):
                out[view_index] = 0.0
            else:
                out[view_index] = (view - mean) / std
        else:
            raise ValueError(f"Unsupported image_normalize={image_normalize!r}")
    return out


def normalize_keep_angles_deg(
    keep_angles_deg: float | Sequence[float] | None,
) -> tuple[float, ...] | None:
    """Normalize a scalar or sequence angle keep-list.

    Used by :class:`DatasetTaskSpec` and :func:`load_role_array` to subset
    multi-view roles to selected acquisition angles.

    Args:
        keep_angles_deg: ``None`` (keep all views), one float, or a non-empty
            sequence of angles.

    Returns:
        ``None`` or a tuple of floats in request order.

    Raises:
        TypeError: If ``keep_angles_deg`` is a boolean.
        ValueError: If a sequence is provided but empty.
    """
    if keep_angles_deg is None:
        return None
    if isinstance(keep_angles_deg, (bool, np.bool_)):
        raise TypeError("keep_angles_deg must be a float or sequence of floats")
    if isinstance(keep_angles_deg, (int, float, np.floating, np.integer)):
        return (float(keep_angles_deg),)
    angles = tuple(float(value) for value in keep_angles_deg)
    if not angles:
        raise ValueError("keep_angles_deg must be non-empty when provided")
    return angles


def row_has_keep_angles(
    row: CatalogRow,
    keep_angles_deg: Sequence[float],
    *,
    atol_deg: float = DEFAULT_ANGLE_ATOL_DEG,
) -> bool:
    """Return True when ``row.angles_deg`` contains every requested angle.

    Matching is approximate within ``atol_deg``; used when building task
    datasets so rows missing a requested view are dropped silently.

    Args:
        row: Catalog sample whose acquisition schedule is checked.
        keep_angles_deg: Angles that must each match some entry in
            ``row.angles_deg``.
        atol_deg: Absolute tolerance in degrees for angle equality.

    Returns:
        ``True`` if every requested angle is present; else ``False``.
    """
    for target in keep_angles_deg:
        if not any(
            abs(float(angle) - float(target)) <= float(atol_deg)
            for angle in row.angles_deg
        ):
            return False
    return True


@dataclass(frozen=True)
class DatasetTaskSpec:
    """Describes one X/Y task over a catalog or schedule-consistent subset.

    ``row_filter`` is simple equality on ``CatalogRow`` fields
    (e.g. ``{"split": "train", "field_status": "complete"}``).
    ``x_fields`` / ``y_fields`` name ``CatalogRow`` attributes; ``RoleRef``
    fields load lazily to ``numpy.ndarray`` with shape ``(V, C, H, W)``.

    ``keep_angles_deg`` optionally subsets multi-view roles to selected
    acquisition angles (e.g. ``180`` or ``(0, 180)``). Rows that do not
    contain every requested angle are dropped. Loaded role arrays then have
    ``V == len(keep_angles_deg)`` in request order.

    ``image_representation`` controls how ``RoleRef`` camera roles resolve:

    - ``"raw_float"`` (default) — float32 linear intensity from ``.raw.tif``
      sidecars (``filenames[role_raw]`` or derived from the display name).
      Supported roles: clean, particle, observed, anomaly (linear
      ``particle - clean``).
    - ``"jpeg_uint8"`` — historical uint8 JPG (or anomaly PNG) via
      ``filenames[role]``. Legacy validation notebooks use this mode.

    ``image_normalize`` optionally remaps loaded **input** (``x_fields``)
    role arrays after loading:

    - ``"none"`` (default) — leave linear intensities unchanged.
    - ``"per_image_minmax"`` — for each view, map to ``[0, 1]`` with
      ``(x - min) / (max - min)`` (diagnostic dynamic-range removal).
    - ``"per_image_zscore"`` — for each view, ``(x - mean) / std``
      (diagnostic; constant views → zeros).
    - ``"train_split_zscore"`` — global ``(x - mean) / std`` using
      ``intensity_mean`` / ``intensity_std`` estimated on the **train**
      split only (preserves cross-sample absolute intensity relationships).

    See also:
        :func:`build_task_dataset` — apply this spec to :class:`CatalogRow` samples.
        :class:`CatalogTaskDataset` — lazy PyTorch dataset over filtered rows.
    """

    name: str
    row_filter: Mapping[str, Any]
    x_fields: tuple[str, ...]
    y_fields: tuple[str, ...]
    image_representation: ImageRepresentation = DEFAULT_IMAGE_REPRESENTATION
    image_normalize: ImageNormalize = DEFAULT_IMAGE_NORMALIZE
    intensity_mean: float | None = None
    intensity_std: float | None = None
    keep_angles_deg: float | Sequence[float] | None = None
    angle_atol_deg: float = DEFAULT_ANGLE_ATOL_DEG

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "keep_angles_deg",
            normalize_keep_angles_deg(self.keep_angles_deg),
        )
        if float(self.angle_atol_deg) < 0.0:
            raise ValueError("angle_atol_deg must be non-negative")
        if self.image_normalize not in SUPPORTED_IMAGE_NORMALIZE:
            raise ValueError(
                f"Unsupported image_normalize={self.image_normalize!r}"
            )
        if self.image_normalize == IMAGE_NORMALIZE_TRAIN_SPLIT_ZSCORE:
            if self.intensity_mean is None or self.intensity_std is None:
                raise ValueError(
                    "train_split_zscore requires intensity_mean and "
                    "intensity_std (estimate on the train split)."
                )
            if float(self.intensity_std) <= 0.0:
                raise ValueError("intensity_std must be positive")
        elif self.intensity_mean is not None or self.intensity_std is not None:
            raise ValueError(
                "intensity_mean / intensity_std are only valid with "
                "image_normalize='train_split_zscore'"
            )

    def intensity_stats(self) -> IntensityStats | None:
        """Return train-split intensity stats when configured, else ``None``."""
        if self.intensity_mean is None or self.intensity_std is None:
            return None
        return IntensityStats(
            mean=float(self.intensity_mean),
            std=float(self.intensity_std),
        )


def _resolve_role_relative_path(
    filenames: Mapping[str, Any],
    role_name: str,
    *,
    image_representation: ImageRepresentation,
    frame_index: int,
    manifest_path: Path,
) -> str:
    """Pick the relative image path for one frame/role under a representation."""
    if image_representation == IMAGE_REPRESENTATION_JPEG_UINT8:
        if role_name not in filenames:
            raise KeyError(
                f"Frame {frame_index} missing filenames[{role_name!r}] "
                f"in {manifest_path}"
            )
        return str(filenames[role_name])

    if image_representation != IMAGE_REPRESENTATION_RAW_FLOAT:
        raise ValueError(
            f"Unsupported image_representation={image_representation!r}; "
            f"expected {IMAGE_REPRESENTATION_RAW_FLOAT!r} or "
            f"{IMAGE_REPRESENTATION_JPEG_UINT8!r}."
        )

    if role_name not in _ROLES_WITH_RAW_FLOAT:
        raise ValueError(
            f"image_representation={IMAGE_REPRESENTATION_RAW_FLOAT!r} is only "
            f"defined for roles {sorted(_ROLES_WITH_RAW_FLOAT)}; got "
            f"{role_name!r}."
        )

    raw_key = f"{role_name}_raw"
    if raw_key in filenames:
        return str(filenames[raw_key])

    if role_name not in filenames:
        raise KeyError(
            f"Frame {frame_index} missing filenames[{role_name!r}] "
            f"(and no {raw_key!r}) in {manifest_path}"
        )
    return role_image_relative_to_raw_tif(str(filenames[role_name]))


def load_role_array(
    role_ref: RoleRef,
    *,
    image_representation: ImageRepresentation = DEFAULT_IMAGE_REPRESENTATION,
    keep_angles_deg: float | Sequence[float] | None = None,
    angle_atol_deg: float = DEFAULT_ANGLE_ATOL_DEG,
) -> np.ndarray:
    """Load one role from ``RoleRef`` alone as ``(V, C, H, W)`` numpy array.

    Default ``image_representation="raw_float"`` reads float32 linear
    intensity from ``.raw.tif`` sidecars. Pass ``"jpeg_uint8"`` for the
    historical JPG (uint8) path.

    When ``keep_angles_deg`` is set, only frames whose ``angle_deg`` matches
    a requested value (within ``angle_atol_deg``) are kept, in the order of
    ``keep_angles_deg`` (not acquisition order).

    Path: manifest -> ordered ``frames[].filenames[...]`` -> PIL ->
    ``numpy.asarray`` (H, W[, C]) -> ``(C, H, W)`` -> stack ``(V, C, H, W)``.
    """
    keep = normalize_keep_angles_deg(keep_angles_deg)
    manifest_path = Path(role_ref.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"RoleRef manifest does not exist: {manifest_path}"
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"RoleRef manifest is not a JSON object: {manifest_path}"
        )

    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(
            f"RoleRef manifest has no frames list: {manifest_path}"
        )

    sequence_dir = manifest_path.parent
    role_name = role_ref.role_name

    def _load_frame(frame_index: int, frame: Mapping[str, Any]) -> np.ndarray:
        filenames = frame.get("filenames")
        if not isinstance(filenames, dict):
            raise KeyError(
                f"Frame {frame_index} missing filenames object in {manifest_path}"
            )
        relative_name = _resolve_role_relative_path(
            filenames,
            role_name,
            image_representation=image_representation,
            frame_index=frame_index,
            manifest_path=manifest_path,
        )
        image_path = sequence_dir / relative_name
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing role image for {role_name!r} "
                f"({image_representation}): {image_path}"
            )

        array = np.asarray(Image.open(image_path))
        if image_representation == IMAGE_REPRESENTATION_RAW_FLOAT:
            array = np.asarray(array, dtype=np.float32)
        if array.ndim == 2:
            array = array[:, :, None]
        elif array.ndim != 3:
            raise ValueError(
                f"Unsupported image rank {array.ndim} for {image_path}"
            )
        return np.transpose(array, (2, 0, 1))

    if keep is None:
        chw_frames: list[np.ndarray] = []
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ValueError(
                    f"Frame {frame_index} is not an object in {manifest_path}"
                )
            chw_frames.append(_load_frame(frame_index, frame))
    else:
        indexed: list[tuple[int, dict[str, Any], float]] = []
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ValueError(
                    f"Frame {frame_index} is not an object in {manifest_path}"
                )
            if "angle_deg" not in frame:
                raise KeyError(
                    f"Frame {frame_index} missing angle_deg in {manifest_path}"
                )
            indexed.append((frame_index, frame, float(frame["angle_deg"])))

        chw_frames = []
        for target in keep:
            matches = [
                (frame_index, frame)
                for frame_index, frame, angle in indexed
                if abs(angle - float(target)) <= float(angle_atol_deg)
            ]
            if not matches:
                raise ValueError(
                    f"No frame with angle_deg≈{float(target)} "
                    f"(atol={float(angle_atol_deg)}) in {manifest_path}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple frames match angle_deg≈{float(target)} "
                    f"in {manifest_path}"
                )
            frame_index, frame = matches[0]
            chw_frames.append(_load_frame(frame_index, frame))

    stacked = np.stack(chw_frames, axis=0)
    if stacked.ndim != 4:
        raise ValueError(
            f"Expected stacked role array rank 4, got shape {stacked.shape}"
        )
    return stacked


def _resolve_field(
    row: CatalogRow,
    field_name: str,
    *,
    image_representation: ImageRepresentation,
    keep_angles_deg: tuple[float, ...] | None,
    angle_atol_deg: float,
) -> Any:
    if not hasattr(row, field_name):
        raise AttributeError(
            f"CatalogRow has no field {field_name!r}"
        )
    value = getattr(row, field_name)
    if isinstance(value, RoleRef):
        return load_role_array(
            value,
            image_representation=image_representation,
            keep_angles_deg=keep_angles_deg,
            angle_atol_deg=angle_atol_deg,
        )
    if value is None and field_name.endswith("_ref"):
        raise ValueError(
            f"CatalogRow.{field_name} is None; cannot load role array "
            f"for sequence_id={row.sequence_id!r}"
        )
    return value


def resolve_task_sample(
    row: CatalogRow,
    task: DatasetTaskSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one catalog row into task ``(x, y)`` field dictionaries.

    Only ``task.x_fields`` / ``task.y_fields`` are returned. ``RoleRef``
    fields load lazily to ``numpy.ndarray`` with shape ``(V, C, H, W)``;
    other fields are copied from the row as scalars / metadata values.
    """
    representation = task.image_representation
    keep = task.keep_angles_deg
    assert keep is None or isinstance(keep, tuple)
    x = {}
    for name in task.x_fields:
        value = _resolve_field(
            row,
            name,
            image_representation=representation,
            keep_angles_deg=keep,
            angle_atol_deg=float(task.angle_atol_deg),
        )
        if isinstance(getattr(row, name, None), RoleRef):
            value = apply_image_normalize(
                value,
                task.image_normalize,
                intensity_stats=task.intensity_stats(),
            )
        x[name] = value
    y = {
        name: _resolve_field(
            row,
            name,
            image_representation=representation,
            keep_angles_deg=keep,
            angle_atol_deg=float(task.angle_atol_deg),
        )
        for name in task.y_fields
    }
    return x, y


def _row_matches_filter(row: CatalogRow, row_filter: Mapping[str, Any]) -> bool:
    """Return True when ``row`` satisfies simple equality ``row_filter``."""
    for key, expected in row_filter.items():
        if not hasattr(row, key):
            raise AttributeError(
                f"row_filter key {key!r} is not a CatalogRow field"
            )
        if getattr(row, key) != expected:
            return False
    return True


def build_task_dataset(
    catalog_rows: list[CatalogRow] | tuple[CatalogRow, ...],
    task: DatasetTaskSpec,
) -> CatalogTaskDataset:
    """Build a lazy task dataset from flat catalog rows and a task spec.

    Applies ``task.row_filter`` with simple field equality, then optionally
    requires ``task.keep_angles_deg`` to be present in each row's
    ``angles_deg``, and returns a ``CatalogTaskDataset`` over the selected
    rows in catalog order.

    See also:
        :class:`CatalogTaskDataset` — returned lazy dataset interface.
        :func:`~tomography_ml.training.training_helpers.make_batch_xy_multiview` — batch adapter for fusion training.
    """
    keep = task.keep_angles_deg
    assert keep is None or isinstance(keep, tuple)
    selected = tuple(
        row
        for row in catalog_rows
        if _row_matches_filter(row, task.row_filter)
        and (
            keep is None
            or row_has_keep_angles(
                row,
                keep,
                atol_deg=float(task.angle_atol_deg),
            )
        )
    )
    return CatalogTaskDataset(rows=selected, task=task)


class CatalogTaskDataset:
    """Minimal dataset interface over filtered catalog rows.

    Stores selected :class:`CatalogRow` objects eagerly. Ordinary row fields
    are available immediately. ``RoleRef``-valued fields resolve lazily and
    load image arrays only when requested through :meth:`__getitem__`.

    Represents one :class:`DatasetTaskSpec` applied to a collection of catalog
    rows. Calling ``dataset[i]`` returns the X (features) and Y (labels) field
    selections defined by that task.

    Image loading defaults to float32 ``.raw.tif`` unless the task sets
    ``image_representation="jpeg_uint8"`` (legacy uint8 path).

    See also:
        :func:`build_task_dataset` — preferred constructor with row filtering.
        :class:`~tomography_ml.gummybear_data_catalog.HierarchicalCameraLightDataset.HierarchicalCameraLightDataset` — full ``[I, V, C, H, W]`` joint groups.
        :class:`~tomography_ml.gummybear_data_catalog.IlluminationOnlyDataset.IlluminationOnlyDataset` — multi-light joint groups.
    """

    _REPR_MAX_ROWS = 20

    def __init__(
        self,
        rows: tuple[CatalogRow, ...],
        task: DatasetTaskSpec,
    ):
        """Wrap pre-filtered catalog rows with a fixed task specification.

        Prefer :func:`build_task_dataset` to apply ``row_filter`` and
        ``keep_angles_deg`` selection. Role images load lazily on
        :meth:`__getitem__`; default representation is float32 ``.raw.tif``.

        Args:
            rows: Selected :class:`CatalogRow` samples in catalog order.
            task: Field selections, filters, and loading options for this task.
        """
        self.rows = rows
        self.task = task

    def __len__(self) -> int:
        """Return the number of catalog samples in this task dataset."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve one catalog sample into task ``(x, y)`` field dicts.

        ``RoleRef`` fields in ``task.x_fields`` load to ``(V, C, H, W)``
        numpy arrays; ``y_fields`` are scalars or metadata from the row.

        Args:
            index: Sample index in ``[0, len(self))``.

        Returns:
            Tuple ``(x, y)`` of dictionaries keyed by ``task.x_fields`` and
            ``task.y_fields``.
        """
        return resolve_task_sample(
            self.rows[index],
            self.task,
        )

    def __repr__(self) -> str:
        """Readable task + sample summary (repo-relative paths; table truncated)."""
        from gummybear.datasets.text_table import format_text_table
        from gummybear.paths import display_path

        task = self.task
        keep = task.keep_angles_deg
        keep_label = "-" if keep is None else ",".join(f"{a:g}" for a in keep)
        view_count = "-" if keep is None else str(len(keep))

        lines = [
            "CatalogTaskDataset(",
            f"  task={task.name!r},",
            f"  n={len(self.rows)},",
            f"  row_filter={dict(task.row_filter)!r},",
            f"  x_fields={task.x_fields!r},",
            f"  y_fields={task.y_fields!r},",
            f"  keep_angles_deg={keep!r},",
            f"  image_representation={task.image_representation!r},",
            f"  image_normalize={task.image_normalize!r},",
        ]

        headers = (
            "i",
            "sequence_id",
            "split",
            "status",
            "V",
            "schedule",
            "px",
            "py",
            "pz",
            "sequence_dir",
        )
        table_rows: list[tuple[str, ...]] = []
        shown = self.rows[: self._REPR_MAX_ROWS]
        for index, row in enumerate(shown):
            v_label = view_count if keep is not None else str(row.frame_count)
            table_rows.append(
                (
                    str(index),
                    row.sequence_id,
                    row.split,
                    row.field_status,
                    v_label,
                    row.camera_schedule_id,
                    "-" if row.particle_x is None else f"{row.particle_x:g}",
                    "-" if row.particle_y is None else f"{row.particle_y:g}",
                    "-" if row.particle_z is None else f"{row.particle_z:g}",
                    display_path(row.sequence_dir),
                )
            )

        if table_rows:
            lines.append("  samples=")
            table = format_text_table(headers, table_rows)
            lines.extend(f"    {line}" for line in table.splitlines())
            remaining = len(self.rows) - len(shown)
            if remaining > 0:
                lines.append(f"    ... ({remaining} more)")
        else:
            lines.append("  samples=(empty),")

        if keep is not None:
            lines.append(f"  note=views subset to angles [{keep_label}]")
        lines.append(")")
        return "\n".join(lines)


def estimate_intensity_stats(
    dataset: CatalogTaskDataset | Sequence[tuple[Mapping[str, Any], Any]],
    x_field: str,
    *,
    std_eps: float = DEFAULT_INTENSITY_STD_EPS,
) -> IntensityStats:
    """Estimate global mean/std over all pixels of ``x_field`` in ``dataset``.

    Call on a train-split dataset built with ``image_normalize="none"``.
    Uses a two-pass mean / population-std estimate for numerical stability.
    """
    if not x_field:
        raise ValueError("x_field must be non-empty")

    total = 0
    sum_ = 0.0
    for sample in dataset:
        images = sample[0] if isinstance(sample, tuple) else sample
        arr = np.asarray(images[x_field], dtype=np.float64)
        total += int(arr.size)
        sum_ += float(arr.sum())
    if total <= 0:
        raise ValueError("cannot estimate intensity stats on an empty dataset")
    mean = sum_ / float(total)

    sum_sq = 0.0
    for sample in dataset:
        images = sample[0] if isinstance(sample, tuple) else sample
        arr = np.asarray(images[x_field], dtype=np.float64)
        sum_sq += float(np.sum((arr - mean) ** 2))
    std = float(np.sqrt(sum_sq / float(total)))
    return IntensityStats(mean=float(mean), std=max(std, float(std_eps)))
