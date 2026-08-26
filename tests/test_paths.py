"""Tests for display-safe path helpers."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

from gummybear.paths import (
    checkpoint_dir,
    display_path,
    display_safe_warnings,
    display_text_paths,
    install_display_safe_warning_paths,
    repo_relative_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_relative_path_under_repo():
    target = REPO_ROOT / "configs" / "m8" / "localization_single_particle.xlsx"
    assert repo_relative_path(target) == "configs/m8/localization_single_particle.xlsx"


def test_checkpoint_dir_creates_milestone_folder(tmp_path: Path):
    out = checkpoint_dir(tmp_path, "m8")
    assert out == tmp_path / "checkpoints" / "m8"
    assert out.is_dir()
    assert checkpoint_dir(tmp_path, "M8") == out


def test_checkpoint_dir_rejects_nested_milestone(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError):
        checkpoint_dir(tmp_path, "m8/extra")


def test_display_path_none_and_empty():
    assert display_path(None) == "-"
    assert display_path("") == "-"


def test_display_path_repo_relative():
    target = REPO_ROOT / "data" / "generated" / "m8_demo"
    assert display_path(target) == "data/generated/m8_demo"


def test_display_path_home_uses_tilde(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    nested = home / "Documents" / "notes.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("x", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    # Force outside-repo absolute rendering by pointing at a path with no
    # pyproject.toml ancestors under home.
    rendered = display_path(nested)
    assert rendered.startswith("~/")
    assert "Documents/notes.txt" in rendered
    assert str(home) not in rendered


def test_display_path_temp_uses_prefix():
    with tempfile.TemporaryDirectory(prefix="gummybear_path_test_") as tmp:
        nested = Path(tmp) / "m8_demo.xlsx"
        nested.write_text("x", encoding="utf-8")
        rendered = display_path(nested)
        assert rendered.startswith("<temp>/")
        assert rendered.endswith("m8_demo.xlsx")
        assert str(Path(tempfile.gettempdir()).resolve()) not in rendered


def test_display_text_paths_rewrites_home_in_log(tmp_path, monkeypatch):
    home = tmp_path / "home"
    conf = home / ".povray" / "3.7" / "povray.conf"
    conf.parent.mkdir(parents=True)
    conf.write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    log = (
        f"povray: cannot open the user configuration file {conf}: "
        "No such file or directory\n"
    )
    filtered = display_text_paths(log)
    assert str(home) not in filtered
    assert "~/.povray/3.7/povray.conf" in filtered


def test_display_safe_warnings_rewrites_filename(tmp_path, monkeypatch):
    home = tmp_path / "home"
    nested = home / "site-packages" / "tqdm" / "auto.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    import pytest

    with pytest.warns(UserWarning) as recorded:
        with display_safe_warnings():
            warnings.warn_explicit(
                "IProgress not found. Please update jupyter and ipywidgets.",
                UserWarning,
                str(nested),
                21,
            )
    filename = str(recorded[0].filename)
    assert filename == display_path(nested)
    assert str(home) not in filename
    assert filename.startswith("~/")
    assert not Path(filename).is_absolute()


def test_install_display_safe_warning_paths_rewrites_showwarning(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    nested = home / "site-packages" / "tqdm" / "auto.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    printed: list[str] = []

    def baseline(message, category, filename, lineno, file=None, line=None):
        printed.append(str(filename))

    monkeypatch.setattr(warnings, "showwarning", baseline)
    install_display_safe_warning_paths()
    install_display_safe_warning_paths()  # idempotent

    warnings.showwarning(
        UserWarning("IProgress not found."),
        UserWarning,
        str(nested),
        21,
    )
    assert printed == [display_path(nested)]
    assert str(home) not in printed[0]
    assert not Path(printed[0]).is_absolute()
