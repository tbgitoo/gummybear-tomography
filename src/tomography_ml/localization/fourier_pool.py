"""Fixed Fourier / low-frequency spatial pooling helpers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def enumerate_fourier_modes(n_modes: int) -> tuple[tuple[int, int, str], ...]:
    """Deterministic low-frequency ``(kx, ky, kind)`` ordering.

    Order by increasing degree ``|kx| + |ky|``. The zero mode is a constant;
    each non-zero ``(kx, ky)`` contributes ``cos`` then ``sin``. Truncate after
    ``n_modes`` entries.

    Args:
        n_modes: Number of modes to retain (must be >= 1).

    Returns:
        Tuple of ``(kx, ky, kind)`` triples with ``kind`` in
        ``{"const", "cos", "sin"}``.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1; got {n_modes}")
    modes: list[tuple[int, int, str]] = []
    modes.append((0, 0, "const"))
    degree = 1
    while len(modes) < n_modes:
        for kx in range(degree, -1, -1):
            ky = degree - kx
            if kx == 0 and ky == 0:
                continue
            modes.append((kx, ky, "cos"))
            if len(modes) >= n_modes:
                break
            modes.append((kx, ky, "sin"))
            if len(modes) >= n_modes:
                break
        degree += 1
    return tuple(modes[:n_modes])


def build_fourier_basis(
    modes: tuple[tuple[int, int, str], ...],
    *,
    height: int,
    width: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Materialise fixed low-frequency 2D basis maps for coded pooling.

    Builds one ``[H, W]`` template per mode on ``[0, 2π) × [0, 2π)``:
    ``const → 1``, ``cos → cos(kx·x + ky·y)``, ``sin → sin(…)``. Modes come
    from ``enumerate_fourier_modes``; ``FourierCodedPool2d`` keeps ``n_modes =
    n_channels`` so each feature channel is pooled against one basis function
    (compact spatial transfer vs global average pooling (GAP) or full Flatten).

    Args:
        modes: Ordered ``(kx, ky, kind)`` triples.
        height: Spatial height ``H`` (must match feature map).
        width: Spatial width ``W``.
        device: Optional torch device for the grid.
        dtype: Optional dtype (default ``float32``).

    Returns:
        Basis stack ``[C, H, W]`` with ``C = len(modes)``.
    """
    if height < 1 or width < 1:
        raise ValueError(f"height/width must be >= 1; got {(height, width)}")
    n_modes = len(modes)
    if n_modes < 1:
        raise ValueError("modes must be non-empty")
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 2.0 * math.pi, steps=height, device=device, dtype=dtype),
        torch.linspace(0.0, 2.0 * math.pi, steps=width, device=device, dtype=dtype),
        indexing="ij",
    )
    basis = torch.empty(
        (n_modes, height, width),
        device=device,
        dtype=dtype if dtype is not None else torch.float32,
    )
    for index, (kx, ky, kind) in enumerate(modes):
        if kind == "const":
            basis[index].fill_(1.0)
            continue
        phase = float(kx) * xx + float(ky) * yy
        if kind == "cos":
            basis[index] = torch.cos(phase)
        elif kind == "sin":
            basis[index] = torch.sin(phase)
        else:
            raise ValueError(f"unknown mode kind {kind!r} at index {index}")
    return basis


class FourierCodedPool2d(nn.Module):
    """Fixed-basis spatial pooling: ``out_c = mean(x_c * basis_c)``.

    Input ``[B, C, H, W]`` → output ``[B, C]``. The spatial basis is a
    **buffer** (not learned), materialised lazily on first forward.

    Typically used with :class:`~tomography_ml.localization.localizer.LocalizerSingleViewFourier`
    and multi-view fusion trunks that reuse the same ``encode_latent`` cut.
    """

    def __init__(self, n_channels: int):
        """Fix a low-frequency mode list and defer basis construction.

        ``n_channels`` sets both the expected input depth and the number of
        Fourier modes (one basis map per channel). The ``[C, H, W]`` basis is
        a registered buffer, built lazily on first ``forward`` at the incoming
        ``(H, W)``.

        Structure::

            modes = enumerate_fourier_modes(n_channels)   # fixed, not learned
            forward: mean(x_c * basis_c) over H, W  →  [B, C]

        Args:
            n_channels: Feature depth ``C`` from ``Encode.out_channels``.
        """
        super().__init__()
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1; got {n_channels}")
        self.n_channels = int(n_channels)
        self.modes = enumerate_fourier_modes(self.n_channels)
        self.register_buffer("_basis", torch.empty(0), persistent=False)
        self._basis_hw: tuple[int, int] | None = None

    @property
    def mode_list(self) -> tuple[tuple[int, int, str], ...]:
        """Fixed ``(kx, ky, kind)`` modes paired one-to-one with channels.

        Logged for reproducibility; order matches
        ``enumerate_fourier_modes(n_channels)``.
        """
        return self.modes

    def _ensure_basis(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            self._basis_hw == (height, width)
            and self._basis.numel() > 0
            and self._basis.device == device
            and self._basis.dtype == dtype
        ):
            return self._basis
        basis = build_fourier_basis(
            self.modes,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        self._basis = basis
        self._basis_hw = (height, width)
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Channel-wise fixed-basis spatial average (Fourier-coded pool).

        Computes ``out[b, c] = mean_{h,w} x[b, c, h, w] * basis[c, h, w]``.
        Replaces global average pooling (GAP) with a compact, position-sensitive
        spatial-frequency summary. Used by the default Fourier localiser and
        as the shared cut for multi-view fusion (before the multilayer
        perceptron (MLP) head).

        Args:
            x: Spatial features ``[B, C, H, W]`` with ``C == n_channels``.

        Returns:
            Pooled vector ``[B, C]``.
        """
        if x.ndim != 4:
            raise ValueError(
                f"FourierCodedPool2d expects [B,C,H,W]; got shape {tuple(x.shape)}"
            )
        _batch, channels, height, width = x.shape
        if channels != self.n_channels:
            raise ValueError(
                f"expected C={self.n_channels} channels; got {channels}"
            )
        basis = self._ensure_basis(
            height, width, device=x.device, dtype=x.dtype
        )
        return (x * basis.unsqueeze(0)).mean(dim=(-2, -1))


__all__ = [
    "FourierCodedPool2d",
    "build_fourier_basis",
    "enumerate_fourier_modes",
]
