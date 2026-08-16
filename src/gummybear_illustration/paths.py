"""Repository and default M8 illustration paths."""

from __future__ import annotations

from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    """Walk parents from ``start`` (or this file) until ``pyproject.toml``."""
    here = Path(start).expanduser().resolve() if start is not None else Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate repository root (pyproject.toml).")


def default_m8_data_root(root: Path | None = None) -> Path:
    """Default generated M8 single-particle corpus directory."""
    base = repo_root() if root is None else Path(root)
    return base / "data" / "generated" / "m8_1" / "single_particle"


def default_cad_dir(root: Path | None = None) -> Path:
    base = repo_root() if root is None else Path(root)
    return base / "cad"


def default_output_pov(root: Path | None = None) -> Path:
    base = repo_root() if root is None else Path(root)
    return base / "outputs" / "pov" / "m8_physical_scene.pov"
