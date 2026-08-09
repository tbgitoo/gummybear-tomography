"""Alternative / comparison single-view localisers.

The default production path is ``LocalizerSingleViewFourier`` in
``localizer.py`` (Fourier-base + multilayer perceptron (MLP)). This module
keeps spatial-blind pooled, high-capacity Flatten, configurable Fourier, and
a Deep Sets-style embedding mean-pool control.

Multi-view fusion and reuse live in ``localize_multiview.py``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from tomography_ml.localization.encoder import Encode
from tomography_ml.localization.fourier_pool import FourierCodedPool2d

FlattenHeadType = Literal["linear", "mlp"]


class LocalizeSingleView(nn.Module):
    """Spatial-blind pooled single-view localiser (negative control).

    Global average pooling (GAP) discards absolute position before predicting
    localisation targets. Useful as a baseline that cannot exploit spatial
    layout in the feature map.

    Contrast production default :class:`~tomography_ml.localization.localizer.LocalizerSingleViewFourier`
    (Fourier-coded spatial readout).
    """

    def __init__(self, encoder: Encode, *, n_outputs: int = 3):
        """Wire a shared ``Encode`` trunk to a coordinate linear (global average pooling (GAP) path).

        The encoder's global average pooling (GAP) inside ``encoder.forward``
        discards absolute position. Multi-view fusion on the non-Fourier path
        reuses ``encode_latent`` (GAP embed + ReLU) as the per-view fusion cut.

        Structure::

            Encode(…, embed_dim) → forward → [B, embed_dim]
              → Linear(embed_dim, n_outputs)

        Args:
            encoder: Pre-built convolutional neural network (CNN) trunk (typically
                from ``make_encode``).
            n_outputs: Target dimension (typically ``3`` for particle
                coordinates (x, y, z)).
        """
        super().__init__()
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1; got {n_outputs}")
        self.encoder = encoder
        self.n_outputs = int(n_outputs)
        self.lin = nn.Linear(self.encoder.lin.out_features, self.n_outputs)

    @property
    def hidden(self) -> int:
        """Locked latent width after ``encode_latent`` (global average pooling (GAP) embedding dim)."""
        return int(self.encoder.lin.out_features)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Non-Fourier latent ``h``: global average pooling (GAP) embed → ReLU.

        ``[B, C, H, W]`` → ``[B, hidden]``. The final coordinate Linear is
        *not* applied. Matches the user-facing cut
        ``CNN → pool → Linear → ReLU → h``. Used as the per-view fusion cut
        on the spatial-blind pooled path.

        Args:
            x: Single-view batch ``[B, in_channels, H, W]``.

        Returns:
            Latent vector ``[B, hidden]``.
        """
        return torch.relu(self.encoder(x))

    def predict_from_latent(self, h: torch.Tensor) -> torch.Tensor:
        """Map a latent to targets. Prefer ``forward`` for the trained path."""
        return self.lin(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global average pooling (GAP) embed then linear map to targets (pooled baseline).

        Args:
            x: Single-view batch ``[B, in_channels, H, W]``.

        Returns:
            Targets ``[B, n_outputs]``.
        """
        embedding = self.encoder(x)
        return self.lin(embedding)


class LocalizeSingleViewFourier(nn.Module):
    """Configurable Fourier-coded single-view localiser.

    Same compact fixed spatial-frequency readout as the default stack but with
    a swappable head and caller-supplied ``Encode``.     For the retained default
    (base convolutional neural network (CNN) + multilayer perceptron (MLP) head), prefer
    :class:`~tomography_ml.localization.localizer.LocalizerSingleViewFourier`.
    """

    def __init__(
        self,
        encoder: Encode,
        *,
        n_outputs: int = 3,
        hidden: int = 128,
        head_type: FlattenHeadType | str = "linear",
    ):
        """Build a configurable Fourier localiser for architecture comparisons.

        Structure (``head_type='mlp'``)::

            Encode → forward_features → FourierCodedPool2d(C)
              → Linear(C, hidden) → ReLU → Linear(hidden, n_outputs)

        Structure (``head_type='linear'``)::

            Encode → forward_features → FourierCodedPool2d(C)
              → Linear(C, n_outputs)

        Args:
            encoder: Convolutional neural network (CNN) trunk; ``out_channels`` sets pool depth.
            n_outputs: Target dimension (typically ``3``).
            hidden: Multilayer perceptron (MLP) hidden width when
                ``head_type='mlp'``.
            head_type: ``'linear'`` (direct map) or ``'mlp'`` (two-layer head).
        """
        super().__init__()
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1; got {n_outputs}")
        mode = str(head_type).strip().lower()
        if mode not in {"linear", "mlp"}:
            raise ValueError(
                f"head_type must be 'linear' or 'mlp'; got {head_type!r}"
            )
        if mode == "mlp" and hidden < 1:
            raise ValueError(f"hidden must be >= 1 for mlp head; got {hidden}")
        self.encoder = encoder
        self.n_outputs = int(n_outputs)
        self.hidden = int(hidden)
        self.head_type = mode
        self.pool = FourierCodedPool2d(self.encoder.out_channels)
        if mode == "linear":
            self.head = nn.Linear(self.encoder.out_channels, self.n_outputs)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.encoder.out_channels, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.n_outputs),
            )

    @property
    def mode_list(self) -> tuple[tuple[int, int, str], ...]:
        """Fixed Fourier ``(kx, ky, kind)`` list from ``FourierCodedPool2d``."""
        return self.pool.mode_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fourier pool on spatial features, then configured head.

        Does not expose ``encode_latent``; multi-view fusion should use
        ``LocalizerSingleViewFourier`` or add an explicit latent cut if this
        ablation trunk is frozen into a fusion stack.

        Args:
            x: Single-view batch ``[B, in_channels, H, W]``.

        Returns:
            Targets ``[B, n_outputs]``.
        """
        features = self.encoder.forward_features(x)
        return self.head(self.pool(features))


class LocalizeSingleViewFlatten(nn.Module):
    """High-capacity Flatten-head localiser (positive control).

    Retains full ``C · H' · W'`` spatial layout — upper bound on
    expressiveness vs compact Fourier readout or spatial-blind pooling.
    Contrast :class:`~tomography_ml.localization.localizer.LocalizerSingleViewFourier`
    (production default) and :class:`LocalizeSingleView` (GAP negative control).

    ``head_type``:
      - ``linear``: Flatten → Linear(n_outputs)
      - ``mlp``: Flatten → Linear(hidden) → ReLU → Linear(n_outputs)
    """

    def __init__(
        self,
        encoder: Encode,
        *,
        n_outputs: int = 3,
        hidden: int = 128,
        head_type: FlattenHeadType | str = "mlp",
    ):
        """Build a Flatten-head positive control.

        ``LazyLinear`` infers input size on first forward from
        ``encoder.flatten_length(H, W)``.

        Structure (``head_type='mlp'``, default)::

            Encode → forward_features → Flatten(1)
              → LazyLinear(hidden) → ReLU → Linear(hidden, n_outputs)

        Structure (``head_type='linear'``)::

            Encode → forward_features → Flatten(1) → LazyLinear(n_outputs)

        Args:
            encoder: CNN trunk; downsample / ``pre_flatten_channels`` control
                vector length.
            n_outputs: Target dimension (typically ``3``).
            hidden: Multilayer perceptron (MLP) hidden width when ``head_type='mlp'``.
            head_type: ``'mlp'`` or ``'linear'``.
        """
        super().__init__()
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1; got {n_outputs}")
        mode = str(head_type).strip().lower()
        if mode not in {"linear", "mlp"}:
            raise ValueError(
                f"head_type must be 'linear' or 'mlp'; got {head_type!r}"
            )
        if mode == "mlp" and hidden < 1:
            raise ValueError(f"hidden must be >= 1 for mlp head; got {hidden}")
        self.encoder = encoder
        self.n_outputs = int(n_outputs)
        self.hidden = int(hidden)
        self.head_type = mode
        self.flat = nn.Flatten(1)
        if mode == "linear":
            self.head = nn.Sequential(nn.LazyLinear(self.n_outputs))
        else:
            self.head = nn.Sequential(
                nn.LazyLinear(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.n_outputs),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten spatial features and apply the configured head.

        Args:
            x: Single-view batch ``[B, in_channels, H, W]``.

        Returns:
            Targets ``[B, n_outputs]``.
        """
        features = self.encoder.forward_features(x)
        return self.head(self.flat(features))


class LocalizeMeanPool(nn.Module):
    """Deep Sets-style mean pool over view embeddings (embedding-mean control).

    Shared convolutional neural network (CNN) encodes each view independently; embeddings are averaged
    over ``V``. Acquisition angles are ignored.

    Not the normative per-angle-expert fusion path — see
    ``ExpertXyzMeanLocalizer`` in ``localize_multiview.py`` (per-angle
    experts → mean of predicted particle coordinates (x, y, z)).
    """

    def __init__(self, encoder: Encode, *, n_outputs: int = 3):
        """Deep Sets-style mean pool over per-view global average pooling (GAP) embeddings.

        Shared ``Encode`` runs on each view independently; embeddings are
        averaged over ``V`` before a single coordinate linear. Angles are
        ignored. Embedding-mean control — not the normative per-angle-expert
        path (see ``ExpertXyzMeanLocalizer`` in ``localize_multiview.py``).

        Structure::

            views [B, V, C, H, W]
              → Encode per view → [B, V, embed_dim]
              → mean over V → [B, embed_dim]
              → Linear(embed_dim, n_outputs)

        Args:
            encoder: Shared view encoder (GAP path).
            n_outputs: Target dimension (typically ``3``).
        """
        super().__init__()
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1; got {n_outputs}")
        self.encoder = encoder
        self.n_outputs = int(n_outputs)
        self.lin = nn.Linear(self.encoder.lin.out_features, self.n_outputs)

    def encode_views(self, views: torch.Tensor) -> torch.Tensor:
        """Run the shared global average pooling (GAP) encoder on each view.

        Args:
            views: ``[B, C, H, W]`` (single view, ``V=1``) or
                ``[B, V, C, H, W]`` (multi-view).

        Returns:
            Per-view embeddings ``[B, V, embed_dim]``.
        """
        if views.ndim == 4:
            views = views.unsqueeze(1)
        if views.ndim != 5:
            raise ValueError(
                "views must have shape [B, C, H, W] or [B, V, C, H, W]; "
                f"got ndim={views.ndim} shape={tuple(views.shape)}"
            )
        batch, n_views, channels, height, width = views.shape
        flat = views.reshape(batch * n_views, channels, height, width)
        embeddings = self.encoder(flat)
        return embeddings.reshape(batch, n_views, -1)

    def forward(
        self,
        views: torch.Tensor,
        angles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean-pool view embeddings, then predict targets.

        ``angles`` is accepted for API symmetry with geometry-aware fusion
        models but is intentionally unused.

        Args:
            views: ``[B, C, H, W]`` or ``[B, V, C, H, W]``.
            angles: Ignored.

        Returns:
            Targets ``[B, n_outputs]``.
        """
        del angles
        embeddings = self.encode_views(views)
        pooled = embeddings.mean(dim=1)
        return self.lin(pooled)


__all__ = [
    "FlattenHeadType",
    "LocalizeMeanPool",
    "LocalizeSingleView",
    "LocalizeSingleViewFlatten",
    "LocalizeSingleViewFourier",
]
