"""Canonical M10 joint dataset: illumination-major ``[I, V, C, H, W]``.

Standalone indexing API matching :class:`CatalogTaskDataset`: ``x, y = dataset[i]``
with lazy image loads via :func:`resolve_task_sample`. Does **not** subclass
``torch.utils.data.Dataset``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from tomography_ml.gummybear_data_catalog.task_dataset import (
    DatasetTaskSpec,
    resolve_task_sample,
)


class HierarchicalCameraLightDataset:
    """One sample = full light × camera grid ``[I, V, C, H, W]`` (canonical M10).

    Indexing returns ``(x, y)`` dicts:

    - ``x[x_field]`` — float32 array ``[I, V, C, H, W]`` (lazy-loaded)
    - ``y`` — scalar labels for ``y_fields`` (from the first light's row)

    Camera / light schedules are dataset attributes (shared across samples),
    not part of each ``(x, y)`` return.

    See also:
        :func:`~tomography_ml.gummybear_data_catalog.IlluminationOnlyDataset.build_illumination_joint_groups`
        :class:`~tomography_ml.localization.localize_multiview.HierarchicalLightThenCameraFusionLocalizer`
    """

    def __init__(
        self,
        groups: Sequence[Mapping[str, Any]],
        *,
        x_field: str,
        y_fields: Sequence[str],
        camera_angles_deg: Sequence[float],
        light_angles_deg: Sequence[float],
        image_normalize: str = "none",
        task_name: str | None = None,
        row_filter: Mapping[str, Any] | None = None,
    ):
        """Build a lazy joint multi-light × multi-camera dataset.

        Each group (from :func:`build_illumination_joint_groups`) must provide
        a complete ``rows_by_light`` map for every angle in ``light_angles_deg``.
        Images load via :func:`resolve_task_sample` with ``keep_angles_deg`` set
        to ``camera_angles_deg``.

        Args:
            groups: Joint particle units with ``rows_by_light`` catalog rows.
            x_field: ``CatalogRow`` attribute naming the input role (e.g.
                ``"anomaly_ref"``).
            y_fields: Scalar label fields copied from the first light's row.
            camera_angles_deg: Camera orbit angles kept per light (``V``).
            light_angles_deg: Illumination angles defining stack order (``I``).
            image_normalize: Normalisation mode for :class:`DatasetTaskSpec`.
            task_name: Optional task label; defaults to a hierarchical name.
            row_filter: Optional filter recorded on the task spec (groups are
                assumed already filtered upstream).
        """
        self.groups = list(groups)
        self.x_field = str(x_field)
        self.y_fields = tuple(str(n) for n in y_fields)
        self.camera_angles_deg = tuple(float(a) for a in camera_angles_deg)
        self.light_angles_deg = tuple(float(a) for a in light_angles_deg)
        if not self.camera_angles_deg:
            raise ValueError("camera_angles_deg must be non-empty")
        if not self.light_angles_deg:
            raise ValueError("light_angles_deg must be non-empty")
        self.n_cameras = len(self.camera_angles_deg)
        self.n_lights = len(self.light_angles_deg)
        name = task_name or "m10_hierarchical_camera_light"
        self.task = DatasetTaskSpec(
            name=name,
            row_filter=dict(row_filter or {}),
            x_fields=(self.x_field,),
            y_fields=self.y_fields,
            keep_angles_deg=list(self.camera_angles_deg),
            image_normalize=image_normalize,
        )

    def __len__(self) -> int:
        """Return the number of joint multi-light particle groups."""
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load one sample: stacked lights × cameras as ``(x, y)``.

        Resolves each light's catalog row lazily; targets are taken from the
        first light in ``light_angles_deg`` (shared particle label across lights).

        Args:
            index: Group index in ``[0, len(self))``.

        Returns:
            Tuple ``(x, y)`` where ``x[x_field]`` is float32 ``[I, V, C, H, W]``
            and ``y`` maps ``y_fields`` to floats.
        """
        group = self.groups[index]
        per_light: list[np.ndarray] = []
        targets: dict[str, float] | None = None
        for light_deg in self.light_angles_deg:
            row = group["rows_by_light"][light_deg]
            images, y = resolve_task_sample(row, self.task)
            v = np.asarray(images[self.x_field], dtype=np.float32)
            if v.ndim == 3:
                v = v[np.newaxis, ...]
            if int(v.shape[0]) != self.n_cameras:
                raise ValueError(
                    f"expected {self.n_cameras} camera views; got {v.shape[0]} "
                    f"for {row.sequence_id}"
                )
            per_light.append(v)
            if targets is None:
                targets = {name: float(y[name]) for name in self.y_fields}
        assert targets is not None
        views = np.stack(per_light, axis=0)  # [I, V, C, H, W]
        return {self.x_field: views}, targets

    def __repr__(self) -> str:
        return (
            f"HierarchicalCameraLightDataset("
            f"n={len(self.groups)}, "
            f"I={self.n_lights}, V={self.n_cameras}, "
            f"x_field={self.x_field!r}, "
            f"task={self.task.name!r})"
        )
