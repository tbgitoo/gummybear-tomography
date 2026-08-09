"""WIN 3A LocalizeSingleViewFlatten shape smoke tests."""

from __future__ import annotations

import torch

from tomography_ml.localization import Encode, LocalizeSingleViewFlatten


def test_flatten_localizer_default_three_outputs() -> None:
    model = LocalizeSingleViewFlatten(Encode())
    out = model(torch.randn(2, 1, 16, 16))
    assert out.shape == (2, 3)


def test_flatten_localizer_n_outputs_and_preserves_grad() -> None:
    model = LocalizeSingleViewFlatten(Encode(), n_outputs=1, hidden=64)
    x = torch.randn(3, 1, 12, 12, requires_grad=True)
    out = model(x)
    assert out.shape == (3, 1)
    out.sum().backward()
    assert x.grad is not None


def test_encode_forward_features_shape() -> None:
    enc = Encode()
    feats = enc.forward_features(torch.randn(2, 1, 20, 24))
    assert feats.shape == (2, 64, 20, 24)
