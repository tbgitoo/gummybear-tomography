"""Parametrised convolutional neural network (CNN) encoder shared by single-view localisers and fusion trunks."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

DownsampleMode = Literal["base", "low", "medium", "high"]

CHANNEL_PRESETS: dict[str, tuple[int, ...]] = {
    "narrow": (8, 16, 32),
    "base": (16, 32, 64),
    "wide": (32, 64, 128),
    "shallow": (16, 32),
    "deeper": (16, 32, 64, 64),
}

# ``base`` = default: no MaxPool. Increasing modes add 2×2 pools.
DOWNSAMPLE_MODES: frozenset[str] = frozenset({"base", "low", "medium", "high"})


def resolve_channels(
    channels: tuple[int, ...] | str | None = None,
) -> tuple[int, ...]:
    """Resolve per-block channel widths for ``Encode``.

    Accepts an explicit tuple or a named preset. Presets: ``narrow``
    ``(8, 16, 32)``, ``base`` ``(16, 32, 64)`` (default), ``wide``
    ``(32, 64, 128)``, ``shallow`` ``(16, 32)``, ``deeper``
    ``(16, 32, 64, 64)``. Used by encoder builders and architecture
    comparison grids; the default Fourier localiser pins ``base``.

    Args:
        channels: Explicit per-block widths, a preset name, or ``None`` for
            ``base``.

    Returns:
        Tuple of positive channel widths, one entry per conv block.
    """
    if channels is None:
        return CHANNEL_PRESETS["base"]
    if isinstance(channels, str):
        key = channels.strip().lower()
        if key not in CHANNEL_PRESETS:
            raise ValueError(
                f"unknown channel preset {channels!r}; "
                f"expected one of {sorted(CHANNEL_PRESETS)}"
            )
        return CHANNEL_PRESETS[key]
    if len(channels) < 1:
        raise ValueError("channels must be non-empty")
    if any(int(c) < 1 for c in channels):
        raise ValueError(f"channel widths must be >= 1; got {channels}")
    return tuple(int(c) for c in channels)


def _normalize_downsample(downsample: str) -> str:
    """Map downsample labels; ``base`` is the no-pool setting."""
    mode = downsample.strip().lower()
    aliases = {
        "none": "base",
        "strong": "high",
    }
    return aliases.get(mode, mode)


def _pool_after_indices(n_blocks: int, downsample: str) -> frozenset[int]:
    """Return conv-block indices after which a 2×2 MaxPool is applied."""
    mode = _normalize_downsample(downsample)
    if mode not in DOWNSAMPLE_MODES:
        raise ValueError(
            f"downsample must be one of {sorted(DOWNSAMPLE_MODES)}; got {downsample!r}"
        )
    if mode == "base" or n_blocks < 1:
        return frozenset()
    if mode == "low":
        return frozenset({0})
    if mode == "medium":
        return frozenset({i for i in (0, 1) if i < n_blocks})
    # high: pool after every block
    return frozenset(range(n_blocks))


def spatial_size_after_encode(
    height: int,
    width: int,
    *,
    n_blocks: int,
    downsample: str = "base",
) -> tuple[int, int]:
    """Predict spatial size ``(H', W')`` after encoder MaxPool stages.

    ``base`` applies no pools; ``low`` / ``medium`` / ``high`` add 2×2
    MaxPool layers after the first one / two / all conv blocks. Each pool
    halves ``H`` and ``W`` via integer floor division. Used by
    ``Encode.feature_map_size`` and high-capacity Flatten-head length
    planning (``flatten_length``).

    Args:
        height: Input feature-map height.
        width: Input feature-map width.
        n_blocks: Number of conv blocks (``len(channels)``).
        downsample: Pool schedule label (``base``, ``low``, ``medium``,
            ``high``; aliases ``none`` → ``base``, ``strong`` → ``high``).

    Returns:
        ``(H', W')`` after all scheduled pools.
    """
    pool_after = _pool_after_indices(n_blocks, downsample)
    h, w = int(height), int(width)
    for i in range(n_blocks):
        if i in pool_after:
            h = h // 2
            w = w // 2
    return h, w


def describe_downsample_schedule(
    n_blocks: int,
    downsample: str,
    *,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    """Human-readable MaxPool schedule for architecture comparison tables.

    The encoder applies optional ``MaxPool2d(2, stride=2)`` **after** selected
    conv blocks (each pool halves ``H`` and ``W``). Mode ``base`` = no pools;
    ``low`` / ``medium`` / ``high`` pool after the first one / two / all blocks.

    Args:
        n_blocks: Number of conv blocks (``len(encoder_channels)``).
        downsample: One of :data:`DOWNSAMPLE_MODES`.
        height: Optional input height for ``spatial_hw_path`` / ``feature_map_hw``.
        width: Optional input width for ``spatial_hw_path`` / ``feature_map_hw``.

    Returns:
        Dict with ``downsample_mode``, ``n_conv_blocks``, ``maxpool_after_blocks``,
        ``pool_schedule``, and when ``height``/``width`` are given also
        ``spatial_hw_path`` and ``feature_map_hw`` ``(H_out, W_out)``.
    """
    n_blocks = int(n_blocks)
    mode = _normalize_downsample(downsample)
    pool_after = _pool_after_indices(n_blocks, downsample)
    if not pool_after:
        maxpool_after = "none"
        pool_schedule = f"{n_blocks}× (Conv3×3 → ReLU), no MaxPool"
    else:
        block_list = ", ".join(str(i) for i in sorted(pool_after))
        maxpool_after = block_list.replace(" ", "")
        pool_schedule = (
            f"{n_blocks}× (Conv3×3 → ReLU); MaxPool2×2 stride 2 after block(s) "
            f"{block_list}"
        )

    out: dict[str, Any] = {
        "downsample_mode": mode,
        "n_conv_blocks": n_blocks,
        "maxpool_after_blocks": maxpool_after,
        "pool_schedule": pool_schedule,
        "spatial_hw_path": None,
        "feature_map_hw": None,
    }
    if height is None or width is None:
        return out

    h, w = int(height), int(width)
    hw_steps = [f"{h}×{w}"]
    for i in range(n_blocks):
        if i in pool_after:
            h, w = h // 2, w // 2
        hw_steps.append(f"{h}×{w}")
    out["spatial_hw_path"] = " → ".join(hw_steps)
    out["feature_map_hw"] = spatial_size_after_encode(
        int(height),
        int(width),
        n_blocks=n_blocks,
        downsample=downsample,
    )
    return out


class Encode(nn.Module):
    """Small convolutional neural network (CNN) trunk for single-view localisation heads.

    Defaults: channel preset ``base`` ``(16, 32, 64)`` with downsample
    ``base`` (no MaxPool). Downstream heads attach after ``forward_features``:
    spatial-blind global average pooling (GAP) baseline, fixed Fourier-coded
    pool (compact spatial readout), or full spatial flatten (high-capacity
    upper bound).

    Raise downsample to ``low`` / ``medium`` / ``high`` to shrink feature-map
    resolution. For Flatten-head length control, prefer ``medium`` or ``high``
    and/or ``pre_flatten_channels``.

    Typically used with :func:`~tomography_ml.localization.builders.make_encode`
    and readout heads from :func:`~tomography_ml.localization.builders.make_fourier`,
    :func:`~tomography_ml.localization.builders.make_pooled`, or
    :func:`~tomography_ml.localization.builders.make_flatten`.
    """

    def __init__(
        self,
        *,
        channels: tuple[int, ...] | str | None = None,
        downsample: DownsampleMode | str = "base",
        pre_flatten_channels: int | None = None,
        embed_dim: int = 128,
        in_channels: int = 1,
    ):
        """Build the shared convolutional neural network (CNN) trunk.

        Each block is ``Conv3×3 → ReLU`` with an optional ``MaxPool2×2`` after
        it (controlled by ``downsample``). An optional ``1×1`` conv can shrink
        channels before Flatten heads. The global average pooling (GAP) path
        (``forward``) ends in
        ``AdaptiveAvgPool2d(1) → Flatten → Linear(out_channels, embed_dim)``;
        Fourier and Flatten localisers call ``forward_features`` instead and
        attach their own pooling or flatten head.

        Structure (defaults)::

            [in] → Conv(1→16,3×3)+ReLU → Conv(16→32,3×3)+ReLU
                 → Conv(32→64,3×3)+ReLU → [C=64, H'=H, W'=W]
            forward: … → AdaptiveAvgPool → Linear(64, embed_dim)
            forward_features: stops before global pool

        Args:
            channels: Per-block widths or preset name (default ``base``).
            downsample: MaxPool schedule; ``base`` = no pools.
            pre_flatten_channels: Optional ``1×1`` channel cap before Flatten
                (reduces flat vector length); ``None`` keeps final block width.
            embed_dim: GAP embedding width (``forward`` output dim; typically
                ``128`` for multi-view fusion on the pooled path).
            in_channels: Input image channels (typically ``1``).
        """
        super().__init__()
        resolved = resolve_channels(channels)
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1; got {in_channels}")
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1; got {embed_dim}")
        if pre_flatten_channels is not None and int(pre_flatten_channels) < 1:
            raise ValueError(
                f"pre_flatten_channels must be >= 1 or None; got {pre_flatten_channels}"
            )

        self.channels = resolved
        self.downsample = _normalize_downsample(str(downsample))
        self.pre_flatten_channels = (
            None if pre_flatten_channels is None else int(pre_flatten_channels)
        )
        self.embed_dim = int(embed_dim)
        self.in_channels = int(in_channels)
        self._pool_after = _pool_after_indices(len(resolved), self.downsample)

        blocks: list[nn.Module] = []
        prev = self.in_channels
        for i, width in enumerate(resolved):
            layers: list[nn.Module] = [
                nn.Conv2d(prev, width, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            if i in self._pool_after:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            blocks.append(nn.Sequential(*layers))
            prev = width
        self.blocks = nn.ModuleList(blocks)

        feat_channels = resolved[-1]
        if self.pre_flatten_channels is not None:
            self.channel_compress: nn.Module | None = nn.Conv2d(
                feat_channels,
                self.pre_flatten_channels,
                kernel_size=1,
            )
            feat_channels = self.pre_flatten_channels
        else:
            self.channel_compress = None

        self.out_channels = int(feat_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten(1)
        self.lin = nn.Linear(self.out_channels, self.embed_dim)

    def feature_map_size(self, height: int, width: int) -> tuple[int, int]:
        """Spatial size after ``forward_features`` for a given input resolution.

        Uses this encoder's ``channels`` length and ``downsample`` schedule.
        Same logic as ``spatial_size_after_encode`` but bound to instance
        hyperparameters.

        Args:
            height: Input image height ``H``.
            width: Input image width ``W``.

        Returns:
            ``(H', W')`` feature-map size.
        """
        return spatial_size_after_encode(
            height,
            width,
            n_blocks=len(self.channels),
            downsample=self.downsample,
        )

    def flatten_length(self, height: int, width: int) -> int:
        """Flat vector length ``C · H' · W'`` for the Flatten head.

        High-capacity upper-bound path: ``LocalizeSingleViewFlatten`` consumes
        ``forward_features`` then ``Flatten(1)``; this count sizes ``LazyLinear``
        and parameter budgets. ``C`` is ``out_channels`` (after optional
        ``pre_flatten_channels`` compress).

        Args:
            height: Input image height ``H``.
            width: Input image width ``W``.

        Returns:
            Number of elements after spatial flatten.
        """
        h_out, w_out = self.feature_map_size(height, width)
        return int(self.out_channels * h_out * w_out)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv blocks (and optional channel compress); keep spatial layout.

        Branch point for Fourier-coded pool (compact fixed spatial-frequency
        readout) and Flatten head (high-capacity upper bound). The spatial-blind
        pooled baseline uses ``forward`` (global average pooling (GAP) + linear
        embed) instead.
        Multi-view fusion trunks start from features produced here when
        attached to a Fourier head.

        Args:
            x: Batch of images ``[B, in_channels, H, W]``.

        Returns:
            Spatial feature map ``[B, out_channels, H', W']``.
        """
        for block in self.blocks:
            x = block(x)
        if self.channel_compress is not None:
            x = self.channel_compress(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pool embed for the spatial-blind pooled baseline.

        ``forward_features → AdaptiveAvgPool2d(1) → Flatten → Linear`` yields
        ``[B, embed_dim]``. Used by ``LocalizeSingleView`` (pooled negative
        control) and ``LocalizeMeanPool``; not the default Fourier localiser.

        Args:
            x: Batch of images ``[B, in_channels, H, W]``.

        Returns:
            Pooled embedding ``[B, embed_dim]``.
        """
        features = self.forward_features(x)
        pooled = self.pool(features)
        flattened = self.flat(pooled)
        return self.lin(flattened)
