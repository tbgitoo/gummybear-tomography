"""Illumination-stack datasets for fixed-camera multi-light localisation.

One sample stacks images across illumination angles at a fixed camera
viewpoint. This is the **10_1 subsample** of the canonical M10 joint grid
``[I, V, C, H, W]``: keep one camera index → ``[I, C, H, W]``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from tomography_ml.gummybear_data_catalog.catalog import CatalogRow
from tomography_ml.gummybear_data_catalog.task_dataset import (
    DatasetTaskSpec,
    resolve_task_sample,
)
from tomography_ml.localization.localize_multiview import (
    light_angle_deg_from_optical_setup_id,
)


def particle_id_from_sequence_id(sequence_id: str) -> str:
    """Extract the particle id suffix from a sequence id.

    Multi-illumination sequences encode the joint particle unit as the
    trailing underscore segment (``…_<particle_id>``). Used when grouping
    catalog rows across lights.

    Args:
        sequence_id: Full sequence identifier from the catalog / manifest.

    Returns:
        Substring after the last ``_``, or the whole id if none.
    """
    return str(sequence_id).rsplit("_", 1)[-1]


def build_illumination_joint_groups(
    catalog_rows: Sequence[CatalogRow],
    *,
    light_angles_deg: Sequence[float],
    complete_only: bool = True,
    min_groups: int = 0,
) -> list[dict[str, Any]]:
    """Group catalog rows into joint particle units with a full light set.

    Each returned group::

        {
            "particle_id": str,
            "split": str,
            "rows_by_light": {light_deg: CatalogRow, …},
        }

    Only particles that have a complete row for every angle in
    ``light_angles_deg`` are kept. Raises ``RuntimeError`` on split
    mismatches or when fewer than ``min_groups`` groups are found
    (``min_groups=0`` disables that check).
    """
    rows = list(catalog_rows)
    if complete_only:
        rows = [r for r in rows if r.field_status == "complete"]

    grouped: dict[str, dict[float, CatalogRow]] = defaultdict(dict)
    split_by_particle: dict[str, str] = {}
    for row in rows:
        pid = particle_id_from_sequence_id(row.sequence_id)
        light = float(light_angle_deg_from_optical_setup_id(row.optical_setup_id))
        grouped[pid][light] = row
        split_by_particle.setdefault(pid, row.split)
        if split_by_particle[pid] != row.split:
            raise RuntimeError(f"split mismatch for particle {pid}")

    required_lights = tuple(float(a) for a in light_angles_deg)
    joint_groups: list[dict[str, Any]] = []
    for pid, by_light in sorted(grouped.items()):
        if all(L in by_light for L in required_lights):
            joint_groups.append(
                {
                    "particle_id": pid,
                    "split": split_by_particle[pid],
                    "rows_by_light": {L: by_light[L] for L in required_lights},
                }
            )

    if int(min_groups) > 0 and len(joint_groups) < int(min_groups):
        raise RuntimeError(
            "Need complete multi-light particle groups before illumination "
            f"fusion; have {len(joint_groups)} (min_groups={min_groups})"
        )
    return joint_groups


def groups_for_split(
    joint_groups: Sequence[Mapping[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    """Filter joint illumination groups to one catalog split.

    Args:
        joint_groups: Output of :func:`build_illumination_joint_groups`.
        split: Workbook split name (``train``, ``validation``, or ``test``).

    Returns:
        Subset of groups whose ``split`` field equals ``split``.
    """
    return [g for g in joint_groups if g["split"] == split]


def count_groups_by_split(
    joint_groups: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Count joint illumination groups per catalog split.

    Args:
        joint_groups: Output of :func:`build_illumination_joint_groups`.

    Returns:
        Mapping ``{split_name: n_groups}``.
    """
    by_split: dict[str, int] = defaultdict(int)
    for g in joint_groups:
        by_split[str(g["split"])] += 1
    return dict(by_split)


class IlluminationOnlyDataset(Dataset):
    """One sample = stacked images over lights at a fixed camera angle.

    Equivalent to ``canonical_joint[i, v_fixed]`` for all illuminations ``i``
    from an ``[I, V, C, H, W]`` grid (10_1 V-subsample).

    Returns ``(views, targets, light_angles)`` where:

    - ``views`` — ``[I, C, H, W]`` float tensor (I = number of lights)
    - ``targets`` — dict of float labels for ``y_fields``
    - ``light_angles`` — ``[I]`` float tensor matching stack order

    See also:
        :func:`build_illumination_joint_groups` — upstream catalog grouping.
        :func:`~tomography_ml.training.training_helpers.train_e2e` — fusion training loop.
        :class:`~tomography_ml.localization.localize_multiview.CompactLatentFusionLocalizer` — common fusion head.
    """

    def __init__(
        self,
        groups: Sequence[Mapping[str, Any]],
        *,
        x_field: str,
        y_fields: Sequence[str],
        fixed_camera_deg: float,
        light_angles_deg: Sequence[float],
        image_normalize: str,
        task_name: str | None = None,
    ):
        """Build a PyTorch dataset over multi-light joint groups.

        Each group (from :func:`build_illumination_joint_groups`) must provide
        a complete ``rows_by_light`` map for every angle in ``light_angles_deg``.
        Images load via :func:`resolve_task_sample` with a single kept camera
        angle; default representation is float32 ``.raw.tif``.

        Args:
            groups: Joint particle units with ``rows_by_light`` catalog rows.
            x_field: ``CatalogRow`` attribute naming the input role (e.g.
                ``"observed_ref"``).
            y_fields: Scalar label fields copied from the first light's row.
            fixed_camera_deg: Camera angle passed to ``keep_angles_deg`` (one
                view per light stack).
            light_angles_deg: Illumination angles defining stack order and
                ``V`` dimension.
            image_normalize: Normalisation mode forwarded to
                :class:`DatasetTaskSpec` (e.g. ``"none"``).
            task_name: Optional task label; defaults to an illumination-style name.
        """
        self.groups = list(groups)
        self.x_field = str(x_field)
        self.y_fields = tuple(str(n) for n in y_fields)
        self.fixed_camera_deg = float(fixed_camera_deg)
        self.light_order = tuple(float(a) for a in light_angles_deg)
        self.n_views = len(self.light_order)
        if self.n_views < 1:
            raise ValueError("light_angles_deg must be non-empty")
        self.schedule_light = torch.tensor(self.light_order, dtype=torch.float32)
        name = task_name or f"m10_1_illum_cam{self.fixed_camera_deg:g}"
        self.task = DatasetTaskSpec(
            name=name,
            row_filter={},
            x_fields=(self.x_field,),
            y_fields=self.y_fields,
            keep_angles_deg=self.fixed_camera_deg,
            image_normalize=image_normalize,
        )

    def __len__(self) -> int:
        """Return the number of joint multi-light particle groups."""
        return len(self.groups)

    def __getitem__(self, index: int):
        """Load one sample: stacked lights at a fixed camera angle.

        Resolves each light's catalog row lazily; targets are taken from the
        first light in ``light_order`` (shared particle label across lights).

        Args:
            index: Group index in ``[0, len(self))``.

        Returns:
            Tuple ``(views, targets, light_angles)`` where ``views`` is
            ``[V, C, H, W]`` float32, ``targets`` maps ``y_fields`` to floats,
            and ``light_angles`` is ``[V]`` float32 in stack order.
        """
        group = self.groups[index]
        blocks = []
        targets = None
        for L in self.light_order:
            row = group["rows_by_light"][L]
            images, y = resolve_task_sample(row, self.task)
            v = torch.as_tensor(images[self.x_field], dtype=torch.float32)
            if v.ndim == 4:
                v = v[0]  # single kept camera → [C,H,W]
            if v.ndim != 3:
                raise ValueError(
                    f"expected single-view [C,H,W]; got {tuple(v.shape)} "
                    f"for {row.sequence_id}"
                )
            blocks.append(v)
            if targets is None:
                targets = {name: float(y[name]) for name in self.y_fields}
        views = torch.stack(blocks, dim=0)  # [I, C, H, W]  (10_1 V-subsample)
        return views, targets, self.schedule_light.clone()
