from pathlib import Path
import sys


def test_import_gummybear():
    import gummybear
    assert gummybear.version == "0.0.1.dev0"


def test_import_submodules_all():
    import gummybear.geometry
    import gummybear.rays
    import gummybear.optics
    import gummybear.particles
    import gummybear.datasets
    import gummybear.models
    import gummybear.export_contracts


def test_pyproject_exists():
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = repo_root / "pyproject.toml"
    assert pyproject.is_file()
