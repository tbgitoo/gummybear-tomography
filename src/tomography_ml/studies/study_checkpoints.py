"""Centralized ML study checkpoint paths and I/O policy.

Checkpoints live under ``<repo>/checkpoints/<milestone>/`` with milestone-style
names (``m08_…``, ``m09_…``). Read/write is intended for ``DATA_MODE=="full"``
only so demo runs cannot overwrite full-scale artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from gummybear.paths import checkpoint_dir, display_path

# --- Canonical filenames (milestone-facing; no WIN demode names) ---------------

M08_LEARNING_RATE_STUDY = "m08_learning_rate_study.pt"
M08_TRAIN_VAL_TEST_Z = "m08_train_val_test_z.pt"
M08_TRAIN_VAL_TEST_XYZ = "m08_train_val_test_xyz.pt"
M08_XYZ_SPLIT_SENSITIVITY = "m08_xyz_split_sensitivity.pt"
M09_FROZEN_FOURIER_FUSION = "m09_frozen_fourier_fusion.pt"
M09_FROZEN_POOLED_FUSION = "m09_frozen_pooled_fusion.pt"
M09_E2E_FOURIER_GEOMETRY_FUSION = "m09_e2e_fourier_geometry_fusion.pt"
M09_E2E_POOLED_GEOMETRY_FUSION = "m09_e2e_pooled_geometry_fusion.pt"
M10_FROZEN_ILLUMINATION_FUSION = "m10_frozen_illumination_fusion.pt"
M10_E2E_ILLUMINATION_FUSION = "m10_e2e_illumination_fusion.pt"
M10_HIERARCHICAL_LIGHT_THEN_CAMERA = "m10_hierarchical_light_then_camera.pt"


@dataclass(frozen=True)
class StudyCheckpointPolicy:
    """Resolved checkpoint directory and load/save flags for one study cell."""

    enabled: bool
    load_existing: bool
    retrain: bool
    directory: Path | None

    @property
    def write(self) -> bool:
        """True when a successful training run should persist a checkpoint."""
        return bool(self.enabled)


def study_checkpoint_policy(
    *,
    repo_root: Path | str,
    milestone: str,
    data_mode: str,
    read_checkpoints: bool,
) -> StudyCheckpointPolicy:
    """Build the checkpoint policy for a notebook study cell.

    Parameters
    ----------
    repo_root:
        Repository root (contains ``checkpoints/``).
    milestone:
        Short milestone folder name (``m8``, ``m9``, ``m10``).
    data_mode:
        Notebook ``DATA_MODE``. Checkpoint I/O is enabled only for ``\"full\"``.
    read_checkpoints:
        Notebook ``READ_CHECKPOINTS``. When true (and enabled), load existing
        artifacts instead of overwriting; when false, retrain and overwrite.
    """
    enabled = str(data_mode).strip().lower() == "full"
    if not enabled:
        return StudyCheckpointPolicy(
            enabled=False,
            load_existing=False,
            retrain=False,
            directory=None,
        )
    directory = checkpoint_dir(repo_root, milestone)
    read = bool(read_checkpoints)
    return StudyCheckpointPolicy(
        enabled=True,
        load_existing=read,
        retrain=not read,
        directory=directory,
    )


def study_results_dir(
    policy: StudyCheckpointPolicy,
    *,
    fallback: Path,
) -> Path:
    """Return the on-disk directory for CSVs / sidecars.

    Full mode uses ``policy.directory`` (under ``checkpoints/<milestone>/``).
    Demo/inspect uses ``fallback`` under the study data root.
    """
    if policy.directory is not None:
        return Path(policy.directory)
    path = Path(fallback)
    path.mkdir(parents=True, exist_ok=True)
    return path


def study_checkpoint_path(
    policy: StudyCheckpointPolicy,
    filename: str,
) -> Path | None:
    """Return ``directory / filename`` when checkpointing is enabled, else ``None``."""
    if policy.directory is None:
        return None
    return Path(policy.directory) / str(filename)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU clone of ``model.state_dict()`` suitable for ``torch.save``."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_study_checkpoint(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` to ``path`` (parents created as needed)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), out)
    return out


def load_study_checkpoint(path: Path) -> dict[str, Any]:
    """Load a study checkpoint mapping from ``path``."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(
            f"checkpoint {display_path(path)} must be a dict payload; "
            f"got {type(blob)!r}"
        )
    return blob


def should_load_checkpoint(
    path: Path | None,
    *,
    load_existing: bool,
    retrain: bool,
) -> bool:
    """True when ``path`` exists and the policy asks to load without retraining."""
    return bool(load_existing and not retrain and path is not None and Path(path).is_file())
