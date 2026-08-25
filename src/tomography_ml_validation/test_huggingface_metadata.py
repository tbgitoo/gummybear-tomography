"""Unit tests for Hugging Face metadata export helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tomography_ml_validation.huggingface_metadata import (
    _embedded_preview,
    _illumination_angle_deg,
    _image_value,
    _preview_frame,
    _records_to_table,
    _role_paths,
    dataset_card_markdown,
)


def test_illumination_angle_from_optical_setup_id():
    assert _illumination_angle_deg("opt_m10_illum_120") == 120.0
    assert _illumination_angle_deg("opt_m8_low_001") is None


def test_preview_frame_prefers_requested_angle():
    manifest = {
        "frames": [
            {"angle_deg": 10.0, "frame_index": 1},
            {"angle_deg": 0.0, "frame_index": 0},
        ]
    }
    frame = _preview_frame(manifest, angle_deg=0.0)
    assert frame["frame_index"] == 0


def test_role_paths_are_repo_relative(tmp_path: Path):
    repo = tmp_path / "repo"
    seq = repo / "data" / "generated" / "m8_1" / "single_particle" / "bear_x"
    observed = seq / "observed"
    observed.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    jpg = observed / "frame.jpg"
    jpg.write_bytes(b"fake")
    frame = {
        "filenames": {
            "observed": "observed/frame.jpg",
            "observed_raw": "observed/frame.raw.tif",
        }
    }
    paths = _role_paths(repo, seq, frame)
    assert paths["observed"] == "data/generated/m8_1/single_particle/bear_x/observed/frame.jpg"
    assert paths["observed_raw"] is None  # missing on disk
    assert _image_value(paths["observed"]) == {
        "bytes": None,
        "path": paths["observed"],
    }


def test_embedded_preview_includes_file_bytes(tmp_path: Path):
    repo = tmp_path / "repo"
    rel = "data/generated/m8_1/single_particle/bear_x/observed/frame.jpg"
    jpg = repo / rel
    jpg.parent.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    jpg.write_bytes(b"\xff\xd8fake-jpeg")
    value = _embedded_preview(repo, rel)
    assert value == {
        "bytes": b"\xff\xd8fake-jpeg",
        "path": rel,
    }


def test_dataset_card_contains_both_configs_and_zip_layout():
    md = dataset_card_markdown(
        counts={
            "m8_1": {"train": 1, "validation": 1, "test": 1},
            "m10_illumination": {"train": 2, "validation": 1, "test": 1},
        }
    )
    assert "config_name: m8_1" in md
    assert "config_name: m10_illumination" in md
    assert "dtype: image" in md
    assert "data/generated/m8_1.zip" in md
    assert "data/generated/m10_illumination.zip" in md
    assert "embedded" in md.lower()


def test_records_to_table_embeds_preview_bytes_and_hf_metadata():
    pytest.importorskip("pyarrow")
    records = [
        {
            "corpus": "m8_1",
            "sequence_id": "s",
            "split": "train",
            "optical_setup_id": "opt_m8_low_001",
            "optical_regime": "low",
            "illumination_angle_deg": None,
            "frame_count": 36,
            "preview_angle_deg": 0.0,
            "preview_frame_index": 0,
            "particle_x": 0.0,
            "particle_y": 0.0,
            "particle_z": 0.0,
            "particle_radius": 1.0,
            "bear_mu_s": 1.0,
            "bear_mu_a": 0.1,
            "sequence_dir": "data/generated/m8_1/single_particle/s",
            "manifest_path": "data/generated/m8_1/single_particle/s/manifest.json",
            "observed": {
                "bytes": b"\xff\xd8fake",
                "path": "data/generated/m8_1/single_particle/s/observed/f.jpg",
            },
            "clean": None,
            "particle": None,
            "anomaly": None,
            "observed_raw_path": None,
            "anomaly_raw_path": None,
        }
    ]
    table = _records_to_table(records)
    observed = table.column("observed")[0].as_py()
    assert observed["bytes"] == b"\xff\xd8fake"
    assert observed["path"].endswith("f.jpg")
    metadata = table.schema.metadata or {}
    hf = metadata.get(b"huggingface") or metadata.get("huggingface")
    assert hf is not None
    text = hf.decode("utf-8") if isinstance(hf, bytes) else hf
    assert '"_type": "Image"' in text
    assert "observed" in text
