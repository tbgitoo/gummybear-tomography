"""Live Hugging Face Hub snapshot download (no cache fallback)."""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gummybear.paths import display_path, display_safe_warnings, install_display_safe_warning_paths

DEFAULT_HUB_DOWNLOAD_TIMEOUT_S = 30.0


class HubDownloadError(RuntimeError):
    """Published Hub snapshot could not be fetched (no local-cache fallback)."""


def download_hub_model_snapshot(
    hub_id: str,
    *,
    revision: str | None = None,
    local_dir: Path | str | None = None,
    timeout_s: float = DEFAULT_HUB_DOWNLOAD_TIMEOUT_S,
) -> Path:
    """Fetch a published Hub model snapshot over the network (no cache fallback).

    Always talks to Hugging Face: ``force_download=True``, Hub cache unused,
    ``local_files_only`` forbidden. Writes into ``local_dir`` or a fresh temp
    directory. Offline mode, missing network, or a hung request raise
    :class:`HubDownloadError` after ``timeout_s`` seconds.
    """
    timeout_s = float(timeout_s)
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0; got {timeout_s}")

    install_display_safe_warning_paths()

    with display_safe_warnings():
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.constants import is_offline_mode
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "huggingface_hub is required for Hub download. "
                'Install with: pip install ".[hf]" -c requirements.txt'
            ) from exc

        if is_offline_mode():
            raise HubDownloadError(
                f"Cannot download {hub_id}: Hugging Face offline mode is enabled "
                "(HF_HUB_OFFLINE). This helper tests the published remote repo and "
                "does not use a local cache."
            )

        dest = (
            Path(local_dir)
            if local_dir is not None
            else Path(tempfile.mkdtemp(prefix="gummybear_hub_"))
        )
        dest.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "repo_id": hub_id,
            "repo_type": "model",
            "local_dir": str(dest),
            "force_download": True,
            "local_files_only": False,
            "etag_timeout": min(timeout_s, 10.0),
        }
        if revision is not None:
            kwargs["revision"] = revision

        def _fetch() -> Path:
            with display_safe_warnings():
                return Path(snapshot_download(**kwargs))

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_fetch)
            try:
                return future.result(timeout=timeout_s)
            except TimeoutError:
                future.cancel()
                raise HubDownloadError(
                    f"Timed out after {timeout_s:g}s downloading {hub_id} from "
                    "Hugging Face. Check network connectivity; this helper does "
                    "not fall back to a local cache."
                ) from None
            except HubDownloadError:
                raise
            except Exception as exc:
                raise HubDownloadError(
                    f"Could not download {hub_id} from Hugging Face: {exc}. "
                    "This helper tests the published remote repo and does not "
                    "fall back to a local cache."
                ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "DEFAULT_HUB_DOWNLOAD_TIMEOUT_S",
    "HubDownloadError",
    "download_hub_model_snapshot",
]
