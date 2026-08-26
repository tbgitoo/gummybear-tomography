"""Portable path helpers for display and notebook-facing summaries."""

from __future__ import annotations

import re
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


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


_ABS_POSIX_PATH = re.compile(r"(?:/(?:\.?[\w+.-]+))+")


def display_text_paths(text: str) -> str:
    """Rewrite absolute POSIX paths in ``text`` with :func:`display_path`.

    Used for subprocess logs (e.g. POV-Ray) so notebook output does not leak
    home-directory or temp roots. Tokens that are not absolute paths are
    left unchanged.
    """
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        return display_path(match.group(0))

    return _ABS_POSIX_PATH.sub(_replace, text)


def install_display_safe_warning_paths() -> None:
    """Patch ``warnings.showwarning`` so filenames use :func:`display_path`.

    Idempotent. Needed for third-party import-time warnings (e.g. tqdm's
    ``IProgress`` notice) that fire outside a :func:`display_safe_warnings`
    block and would otherwise print absolute venv / home paths in notebooks.
    Only absolute filenames are rewritten (already-safe ``~/...`` / relative
    paths are left unchanged).
    """
    current = warnings.showwarning
    if getattr(current, "_gummybear_display_safe", False):
        return

    def showwarning(message, category, filename, lineno, file=None, line=None):
        if filename and Path(str(filename)).is_absolute():
            filename = display_path(filename)
        return current(message, category, filename, lineno, file=file, line=line)

    showwarning._gummybear_display_safe = True  # type: ignore[attr-defined]
    warnings.showwarning = showwarning  # type: ignore[assignment]


@contextmanager
def display_safe_warnings() -> Iterator[None]:
    """Re-emit warnings with filenames rewritten through :func:`display_path`.

    Also installs :func:`install_display_safe_warning_paths` so any warning
    that escapes capture (or is printed by a custom ``showwarning``) still
    uses a display-safe filename. Captures warnings issued inside the block
    (including third-party ones such as ``TqdmWarning``) and re-emits them
    with rewritten locations.
    """
    install_display_safe_warning_paths()
    caught: list = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
    finally:
        for item in caught:
            raw = item.filename or ""
            filename = (
                display_path(raw) if raw and Path(str(raw)).is_absolute() else raw
            )
            warnings.warn_explicit(
                str(item.message),
                item.category,
                filename,
                int(item.lineno or 0),
            )
