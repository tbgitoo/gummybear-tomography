"""Tests for display-safe path helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from gummybear.paths import display_path, repo_relative_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_relative_path_under_repo():
    target = REPO_ROOT / "configs" / "m8" / "localization_single_particle.xlsx"
    assert repo_relative_path(target) == "configs/m8/localization_single_particle.xlsx"


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
