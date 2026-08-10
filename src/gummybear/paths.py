"""Portable path helpers for display and notebook-facing summaries."""

from __future__ import annotations

import tempfile
from pathlib import Path


def repo_relative_path(path: str | Path) -> str:
    """Return a repository-relative POSIX path when possible.

    Walks parents looking for ``pyproject.toml``. If none is found, returns a
    POSIX rendering of the resolved path (or the original path on resolve
    failure).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to express relative to the repository root.

    Returns
    -------
    str
        Repository-relative POSIX path when a root is found; otherwise the
        resolved absolute path as POSIX.
    """
    raw = Path(path)
    try:
        resolved = raw.expanduser().resolve()
    except OSError:
        return raw.as_posix()

    for parent in (resolved, *resolved.parents):
        if (parent / "pyproject.toml").is_file():
            try:
                return resolved.relative_to(parent).as_posix()
            except ValueError:
                break
    return resolved.as_posix()


def checkpoint_dir(repo_root: str | Path, milestone: str) -> Path:
    """Return ``<repo_root>/checkpoints/<milestone>/`` for ML study artifacts.

    Example: ``checkpoint_dir(ROOT, "m8") / "m08_learning_rate_study.pt"``.
    """
    name = str(milestone).strip().lower().lstrip("/")
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"milestone must be a simple directory name; got {milestone!r}")
    path = Path(repo_root) / "checkpoints" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_path(path: str | Path | None) -> str:
    """Return a path safe for logs, reprs, and notebook output.

    Prefer a repository-relative path. When the path lies outside the repo:

    - under the user home directory → render with a ``~`` prefix so usernames
      are not leaked;
    - under the process temp directory → render as ``<temp>/...`` so absolute
      system temp roots (e.g. ``/var/folders/...``) are not leaked;
    - otherwise return the absolute path as a last resort.

    Parameters
    ----------
    path : str, pathlib.Path, or None
        Path to format. ``None`` and empty strings render as ``"-"``.

    Returns
    -------
    str
        Display-safe path string.
    """
    if path is None or path == "":
        return "-"
    text = repo_relative_path(path)
    if not Path(text).is_absolute():
        return text
    try:
        resolved = Path(text).expanduser().resolve()
    except OSError:
        return text

    try:
        return ("~" / resolved.relative_to(Path.home().resolve())).as_posix()
    except ValueError:
        pass

    try:
        tmp_root = Path(tempfile.gettempdir()).resolve()
        return f"<temp>/{resolved.relative_to(tmp_root).as_posix()}"
    except ValueError:
        return text
