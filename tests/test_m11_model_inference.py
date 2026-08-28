"""Tests for Hub Fourier model inference helpers (offline via local export dir)."""

from __future__ import annotations

import time
from pathlib import Path

import huggingface_hub
import huggingface_hub.constants as hf_constants
import pytest

from tomography_ml_huggingface.model_export import (
    export_singleview_cnn_fourier,
)
from tomography_ml_huggingface.model_inference import (
    HubDownloadError,
    download_singleview_cnn_fourier,
    load_packaged_m8_demo_example,
    load_singleview_cnn_fourier,
    run_packaged_demo_inference,
)


REPO = Path(__file__).resolve().parents[1]
STUDY_CKPT = REPO / "checkpoints" / "m8" / "m08_train_val_test_xyz.pt"


@pytest.mark.skipif(not STUDY_CKPT.is_file(), reason="M8 xyz study checkpoint absent")
def test_inference_on_packaged_demo_from_exported_weights(tmp_path: Path) -> None:
    local_toml = tmp_path / "local.toml"
    clone = tmp_path / "singleview_cnn_fourier"
    local_toml.write_text(
        "\n".join(
            [
                "[models.singleview_cnn_fourier]",
                'hub_id = "tbhugging/singleview_cnn_fourier"',
                'hub_url = "https://huggingface.co/tbhugging/singleview_cnn_fourier"',
                f'local_clone = "{clone}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    export_singleview_cnn_fourier(
        REPO, local_toml=local_toml, checkpoint_path=STUDY_CKPT
    )
    loaded = load_singleview_cnn_fourier(clone)
    sample = load_packaged_m8_demo_example()
    assert sample.views_chw.shape[0] == 1
    assert sample.views_chw.ndim == 4
    result = run_packaged_demo_inference(loaded)
    assert result.y_true is not None
    assert result.euclidean_error is not None
    assert len(result.y_pred) == 3
    # Sanity: error should be finite and not absurd on the tiny demo corpus.
    assert result.euclidean_error < 50.0


def _patch_live_hub(monkeypatch, snapshot_download, *, offline: bool = False) -> None:
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(hf_constants, "is_offline_mode", lambda: offline)


def test_download_forces_remote_and_skips_hub_cache(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        dest = Path(kwargs["local_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        return str(dest)

    _patch_live_hub(monkeypatch, fake_snapshot_download)
    dest = tmp_path / "snap"
    out = download_singleview_cnn_fourier(
        hub_id="tbhugging/singleview_cnn_fourier", local_dir=dest
    )
    assert out == dest
    assert captured["force_download"] is True
    assert captured["local_files_only"] is False
    assert captured["local_dir"] == str(dest)
    assert "cache_dir" not in captured


def test_download_rejects_offline_mode(monkeypatch, tmp_path: Path) -> None:
    def fake_snapshot_download(**kwargs):
        raise AssertionError("snapshot_download must not run in offline mode")

    _patch_live_hub(monkeypatch, fake_snapshot_download, offline=True)
    with pytest.raises(HubDownloadError, match="offline mode"):
        download_singleview_cnn_fourier(local_dir=tmp_path / "snap")


def test_download_timeout_raises_explicit_error(monkeypatch, tmp_path: Path) -> None:
    def fake_snapshot_download(**kwargs):
        time.sleep(2.0)
        return kwargs["local_dir"]

    _patch_live_hub(monkeypatch, fake_snapshot_download)
    with pytest.raises(HubDownloadError, match="Timed out after 0.2s"):
        download_singleview_cnn_fourier(local_dir=tmp_path / "snap", timeout_s=0.2)


def test_download_connection_error_is_explicit(monkeypatch, tmp_path: Path) -> None:
    def fake_snapshot_download(**kwargs):
        raise ConnectionError("Name or service not known")

    _patch_live_hub(monkeypatch, fake_snapshot_download)
    with pytest.raises(HubDownloadError, match="does not fall back to a local cache"):
        download_singleview_cnn_fourier(local_dir=tmp_path / "snap")


def test_download_rejects_nonpositive_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        download_singleview_cnn_fourier(local_dir=tmp_path / "snap", timeout_s=0)
