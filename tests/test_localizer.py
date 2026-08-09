"""Default LocalizerSingleViewFourier (Fourier-base + MLP) smoke tests."""

from __future__ import annotations

import torch

from tomography_ml.localization import LocalizerSingleViewFourier, count_parameters


def test_localizer_single_view_fourier_default_forward_shape() -> None:
    model = LocalizerSingleViewFourier()
    out = model(torch.randn(4, 1, 32, 32))
    assert out.shape == (4, 3)
    assert len(model.mode_list) == model.encoder.out_channels
    assert model.encoder.channels == (16, 32, 64)
    assert model.encoder.downsample == "base"
    assert isinstance(model.head, torch.nn.Sequential)
    assert count_parameters(model) < 100_000


def test_localizer_single_view_fourier_n_outputs_and_backward() -> None:
    model = LocalizerSingleViewFourier(n_outputs=1, hidden=64)
    x = torch.randn(2, 1, 16, 16, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 1)
    out.sum().backward()
    assert x.grad is not None
