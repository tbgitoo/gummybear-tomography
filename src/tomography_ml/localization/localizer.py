"""Default single-view localiser: compact Fourier readout + multilayer perceptron (MLP) head.

Retained architecture (**Fourier-base + MLP**):

```text
Encode channels (16, 32, 64), downsample=base (no MaxPool)
  → fixed Fourier-coded pool  [B,C,H,W] → [B,C]
  → Linear(C, 128) → ReLU → Linear(128, n_outputs)
```

Alternative architectures (spatial-blind pooled, high-capacity Flatten,
configurable Fourier, embedding mean-pool) live in
``alternative_localizer.py``. Multi-view fusion lives in
``localize_multiview.py``. Fourier basis helpers live in ``fourier_pool.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tomography_ml.localization.encoder import Encode
from tomography_ml.localization.fourier_pool import FourierCodedPool2d

# Default geometry: channel preset "base", downsample "base" (no MaxPool).
_DEFAULT_CHANNELS: tuple[int, ...] = (16, 32, 64)
_DEFAULT_HIDDEN = 128


class LocalizerSingleViewFourier(nn.Module):
    """Default single-view localiser: Fourier-base convolutional neural network (CNN) + multilayer perceptron (MLP) head.

    Predicts localisation targets from one camera view using a compact
    fixed spatial-frequency readout after a small CNN trunk. Distinct from
    the configurable ablation class ``LocalizeSingleViewFourier`` in
    ``alternative_localizer.py``.

    ``n_outputs`` should match ``len(DatasetTaskSpec.y_fields)``
    (e.g. 3 for ``particle_x, particle_y, particle_z``).

    See also:
        :func:`~tomography_ml.localization.builders.m8_single_view_block_freeze` — M8 single-view block freeze.
        :meth:`encode_latent` — fusion cut consumed by compact / geometry-aware heads.
    """

    def __init__(
        self,
        *,
        n_outputs: int = 3,
        hidden: int = _DEFAULT_HIDDEN,
        in_channels: int = 1,
    ):
        """Build the default single-view Fourier localiser stack.

        Fixed convolutional neural network (CNN) geometry with a Fourier-coded
        spatial pool and two-layer multilayer perceptron (MLP) head.
        Multi-view fusion consumes ``encode_latent`` (first MLP
        projection + ReLU, ``hidden`` wide) rather than raw coordinates.

        Structure::

            Encode(channels=(16,32,64), downsample=base, in_channels)
              → forward_features  [B, 64, H, W]
              → FourierCodedPool2d(64)  [B, 64]
              → Linear(64, hidden) → ReLU → Linear(hidden, n_outputs)

        Args:
            n_outputs: Target dimension (typically ``3`` for particle
                coordinates (x, y, z)).
            hidden: MLP hidden width; ``encode_latent`` cut is ``[B, hidden]``
                (default ``128``).
            in_channels: Input image channels (typically ``1``).

        Notebook / protocol: M8 single-view block freeze; used by multi-view fusion trunks.
        """
        super().__init__()
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1; got {n_outputs}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1; got {hidden}")

        self.n_outputs = int(n_outputs)
        self.hidden = int(hidden)
        self.encoder = Encode(
            channels=_DEFAULT_CHANNELS,
            downsample="base",
            in_channels=in_channels,
        )
        self.pool = FourierCodedPool2d(self.encoder.out_channels)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_channels, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.n_outputs),
        )

    @property
    def mode_list(self) -> tuple[tuple[int, int, str], ...]:
        """Fixed Fourier ``(kx, ky, kind)`` list from ``FourierCodedPool2d``.

        One mode per encoder output channel; same ordering as
        ``enumerate_fourier_modes(out_channels)``. Useful for logging and
        cross-run reproducibility checks.
        """
        return self.pool.mode_list

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Shared latent ``h``: after first multilayer perceptron (MLP) projection + ReLU.

        ``[B, C, H, W]`` → ``[B, hidden]``. The final
        ``Linear(hidden → n_outputs)`` is *not* applied. This is the cut
        consumed by multi-view fusion trunks.

        Args:
            x: Single-view batch ``[B, in_channels, H, W]``.

        Returns:
            Latent vector ``[B, hidden]``.

        See also:
            :class:`~tomography_ml.localization.localize_multiview.CompactLatentFusionLocalizer` — ordered-concat fusion over ``h_i``.
            :class:`~tomography_ml.localization.localize_multiview.GeometryAwareFourierFusionLocalizer` — sin/cos geometry tokens.
        """
        pooled = self.pool(self.encoder.forward_features(x))
        return self.head[1](self.head[0](pooled))

    def predict_from_latent(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the final coordinate Linear: ``[B, hidden]`` → ``[B, n_outputs]``."""
        return self.head[2](h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict localisation targets from a single camera view.

        Full path: ``encode_latent`` (Fourier pool + first MLP + ReLU) then
        final ``Linear(hidden → n_outputs)``. For multi-view fusion, call
        ``encode_latent`` on each view and fuse latents upstream.

        Args:
            x: Single-view batch ``[B, in_channels, H, W]``.

        Returns:
            Targets ``[B, n_outputs]`` (e.g. ``[B, 3]`` for xyz coordinates).
        """
        return self.predict_from_latent(self.encode_latent(x))


__all__ = ["LocalizerSingleViewFourier"]
