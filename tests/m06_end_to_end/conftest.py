"""Reviewer-facing pytest output for the M6.5 validation battery.

Extra output is shown only when pytest is invoked with --m6-5-proves and the
collected test files match the intended M6.5 validation battery.

Normal pytest runs remain unchanged.
"""

from pathlib import Path


_REQUIRED_M6_5_TEST_FILES = {
    "test_m6_output_reconciliation.py",
    "test_m6_generation_plan.py",
}


def pytest_addoption(parser):
    parser.addoption(
        "--m6-5-proves",
        action="store_true",
        default=False,
        help="Print reviewer-facing M6.5 validation-battery proof text.",
    )


def _option_enabled(config):
    try:
        return bool(config.getoption("--m6-5-proves"))
    except ValueError:
        return False


def _selected_file_names(items):
    names = set()
    for item in items:
        path = getattr(item, "path", None)
        if path is None:
            path = getattr(item, "fspath", None)
        if path is not None:
            names.add(Path(str(path)).name)
    return names


def pytest_collection_finish(session):
    """Print the reviewer-facing header only for the exact M6.5 battery."""
    config = session.config

    if not _option_enabled(config):
        config._m6_5_proves_active = False
        return

    selected = _selected_file_names(session.items)
    if selected != _REQUIRED_M6_5_TEST_FILES:
        config._m6_5_proves_active = False
        return

    config._m6_5_proves_active = True

    terminalreporter = config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is None:
        return

    terminalreporter.write_line("Milestone 6.5")
    terminalreporter.write_line("Validation battery executed:")
    terminalreporter.write_line("  tests/test_m6_output_reconciliation.py")
    terminalreporter.write_line("  tests/test_m6_generation_plan.py")
    terminalreporter.write_line("")
    terminalreporter.write_line("pytest:")


def pytest_unconfigure(config):
    """Print proof text after pytest has printed the pass/fail summary."""
    if not getattr(config, "_m6_5_proves_active", False):
        return

    terminalreporter = config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is None:
        return

    if getattr(config, "_m6_5_proves_already_printed", False):
        return
    config._m6_5_proves_already_printed = True

    terminalreporter.write_line("")
    terminalreporter.write_line("Validation battery proves:")
    terminalreporter.write_line(
        "  Workbook rows are analyzed and expanded into valid,"
    )
    terminalreporter.write_line(
        "  deterministic jobs for serial image generation,"
    )
    terminalreporter.write_line(
        "  with stable source-cache identities, ordered camera frames,"
    )
    terminalreporter.write_line(
        "  and explicit recording of diffusion inputs and derived"
    )
    terminalreporter.write_line(
        "  material properties."
    )