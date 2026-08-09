"""LocalizeSingleView n_outputs matches DatasetTaskSpec y_fields length."""

from __future__ import annotations

import torch

from tomography_ml.localization import Encode, LocalizeSingleView


def test_single_view_default_three_outputs() -> None:
    model = LocalizeSingleView(Encode())
    assert model(torch.randn(2, 1, 8, 8)).shape == (2, 3)


def test_single_view_n_outputs_one_and_two() -> None:
    x = torch.randn(4, 1, 8, 8)
    assert LocalizeSingleView(Encode(), n_outputs=1)(x).shape == (4, 1)
    assert LocalizeSingleView(Encode(), n_outputs=2)(x).shape == (4, 2)
