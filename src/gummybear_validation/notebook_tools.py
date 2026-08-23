"""Helpers for running installed validation tests from notebooks."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any


def _installed_test_file(test_module: ModuleType) -> Path:
    if test_module.__file__ is None:
        raise ValueError(
            f"Module {test_module.__name__!r} has no __file__; "
            "cannot construct installed pytest path."
        )
    return Path(test_module.__file__).resolve()


def _installed_pytest_ini(test_module: ModuleType) -> Path:
    """Resolve packaged ``pytest.ini`` next to the top-level validation package."""
    top_name = test_module.__name__.split(".", 1)[0]
    top_mod = importlib.import_module(top_name)
    if top_mod.__file__ is None:
        raise FileNotFoundError(
            f"Could not locate a packaged pytest.ini for {top_name!r}."
        )
    candidate = Path(top_mod.__file__).resolve().parent / "pytest.ini"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Missing pytest.ini next to installed package {top_name!r} "
        f"(looked for {candidate})."
    )


def _get_pytest_marks(test_func: Any) -> list:
    marks = getattr(test_func, "pytestmark", [])
    if marks is None:
        return []
    if isinstance(marks, list):
        return marks
    return [marks]


def _get_first_mark_argument(test_func: Any, mark_name: str) -> str | None:
    for mark in _get_pytest_marks(test_func):
        if getattr(mark, "name", None) != mark_name:
            continue
        args = getattr(mark, "args", ())
        if args:
            return str(args[0])
    return None


def _get_test_function(test_module: ModuleType, test_name: str) -> Any:
    if not hasattr(test_module, test_name):
        available = sorted(name for name in dir(test_module) if name.startswith("test_"))
        raise AttributeError(
            f"Test {test_name!r} was not found in module "
            f"{test_module.__name__!r}. Available tests: {available}"
        )
    return getattr(test_module, test_name)


def _core_pytest_lines(pytest_stdout: str) -> list[str]:
    core_lines: list[str] = []
    summary_re = re.compile(
        r"=+\s+.*\b("
        r"passed|failed|error|errors|skipped|warnings|warning|xfailed|xpassed"
        r")\b.*=+",
        flags=re.IGNORECASE,
    )
    progress_re = re.compile(r".*\.py(::[^\s]+)?\s+.*\[\s*\d+%\s*\].*")
    explicit_result_tokens = (
        " PASSED",
        " FAILED",
        " ERROR",
        " SKIPPED",
        " XFAIL",
        " XPASS",
    )
    warning_summary_re = re.compile(r"=+\s+warnings summary\s+=+", flags=re.IGNORECASE)
    in_warning_summary = False

    for line in pytest_stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if warning_summary_re.search(stripped):
            in_warning_summary = True
            core_lines.append(line)
            continue
        if in_warning_summary:
            core_lines.append(line)
            if summary_re.search(stripped):
                in_warning_summary = False
            continue
        if progress_re.match(stripped):
            core_lines.append(line)
            continue
        if any(token in line for token in explicit_result_tokens):
            core_lines.append(line)
            continue
        if summary_re.search(stripped):
            core_lines.append(line)
            continue
    return core_lines


def run_installed_pytest_test(
    test_module: ModuleType,
    test_name: str,
    *,
    show_paths: bool = False,
    show_full_pytest_output: bool = False,
    traceback_style: str = "short",
) -> None:
    """Run one pytest test from an installed validation module (notebook-friendly).

    Resolves the test file from the imported module so notebooks exercise the
    installed package, not an arbitrary source checkout.
    """
    test_func = _get_test_function(test_module, test_name)
    milestone = _get_first_mark_argument(test_func, "milestone")
    proves = _get_first_mark_argument(test_func, "proves")

    test_file = _installed_test_file(test_module)
    pytest_ini = _installed_pytest_ini(test_module)
    test_node = f"{test_file}::{test_name}"

    if milestone is not None:
        print(milestone)
    print(f"Test executed: {test_name}()")

    if show_paths:
        print("\nInstalled pytest node:")
        print(test_node)
        print("\nInstalled pytest config:")
        print(pytest_ini)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(pytest_ini),
        test_node,
        "--color=yes",
        f"--tb={traceback_style}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if show_full_pytest_output:
        if result.stdout:
            print("\npytest stdout:")
            print(result.stdout)
        if result.stderr:
            print("\npytest stderr:")
            print(result.stderr)
    else:
        core_lines = _core_pytest_lines(result.stdout)
        if core_lines:
            print("\npytest:")
            print("\n".join(core_lines))

    label = proves if proves is not None else test_name
    print()
    print(
        textwrap.fill(
            f"Test proves: {label}",
            width=96,
            subsequent_indent=" " * len("Test proves: "),
        )
    )

    if result.returncode != 0:
        if not show_full_pytest_output:
            if result.stdout:
                print("\npytest stdout:")
                print(result.stdout)
            if result.stderr:
                print("\npytest stderr:")
                print(result.stderr)
        raise RuntimeError(
            f"Installed pytest test failed: {test_module.__name__}.{test_name} "
            f"(exit code {result.returncode})"
        )


def run_installed_pytest_tests(
    test_module: ModuleType,
    test_names: tuple[str, ...] | list[str],
    *,
    show_paths: bool = False,
    show_full_pytest_output: bool = False,
    traceback_style: str = "short",
) -> None:
    """Run several installed validation tests sequentially (notebook-friendly)."""
    for test_name in test_names:
        run_installed_pytest_test(
            test_module,
            test_name,
            show_paths=show_paths,
            show_full_pytest_output=show_full_pytest_output,
            traceback_style=traceback_style,
        )
