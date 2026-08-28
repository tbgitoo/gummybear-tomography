"""Machine-local Hugging Face model clone path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gummybear.paths import display_path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover — Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_LOCAL_TOML = Path("configs/hf/local.toml")


@dataclass(frozen=True)
class HfModelLocalPaths:
    """Resolved Hub id + local clone path from ``configs/hf/local.toml``."""

    hub_id: str
    hub_url: str
    local_clone: Path


def load_hf_local_toml(path: Path) -> dict[str, Any]:
    """Parse a machine-local Hugging Face staging TOML."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {display_path(path)}. Copy configs/hf/local.toml.example "
            "to configs/hf/local.toml and set the model's local_clone path."
        )
    with path.open("rb") as fh:
        return tomllib.load(fh)


def resolve_hf_model_paths(
    repo_root: Path,
    *,
    model_key: str,
    local_toml: Path | None = None,
) -> HfModelLocalPaths:
    """Read ``models.<model_key>`` from gitignored ``configs/hf/local.toml``."""
    repo_root = Path(repo_root).resolve()
    toml_path = (
        Path(local_toml) if local_toml is not None else repo_root / DEFAULT_LOCAL_TOML
    )
    if not toml_path.is_absolute():
        toml_path = repo_root / toml_path
    data = load_hf_local_toml(toml_path)
    models = data.get("models") or {}
    row = models.get(model_key)
    if not isinstance(row, Mapping):
        raise KeyError(
            f"{display_path(toml_path)} missing [models.{model_key}] table"
        )
    hub_id = str(row.get("hub_id") or "").strip()
    hub_url = str(row.get("hub_url") or "").strip()
    local_clone = Path(str(row.get("local_clone") or "").strip()).expanduser()
    if not hub_id:
        raise ValueError(f"[models.{model_key}] hub_id is required")
    if not local_clone.is_absolute():
        local_clone = (repo_root / local_clone).resolve()
    if not hub_url:
        hub_url = f"https://huggingface.co/{hub_id}"
    return HfModelLocalPaths(
        hub_id=hub_id, hub_url=hub_url, local_clone=local_clone
    )


__all__ = [
    "DEFAULT_LOCAL_TOML",
    "HfModelLocalPaths",
    "load_hf_local_toml",
    "resolve_hf_model_paths",
]
