"""LocalizeMeanPool shape / angle-agnostic smoke tests (embedding-mean control)."""

from __future__ import annotations

import torch

from tomography_ml.localization import Encode, LocalizeMeanPool


def test_mean_pool_accepts_multi_view_batch() -> None:
    model = LocalizeMeanPool(Encode())
    views = torch.randn(2, 4, 1, 16, 16)
    out = model(views)
    assert out.shape == (2, 3)


def test_mean_pool_n_outputs_matches_head() -> None:
    model = LocalizeMeanPool(Encode(), n_outputs=1)
    views = torch.randn(2, 3, 1, 8, 8)
    assert model(views).shape == (2, 1)

    model2 = LocalizeMeanPool(Encode(), n_outputs=2)
    assert model2(views).shape == (2, 2)


def test_mean_pool_accepts_single_view_as_4d() -> None:
    model = LocalizeMeanPool(Encode())
    views = torch.randn(3, 1, 16, 16)
    out = model(views)
    assert out.shape == (3, 3)


def test_mean_pool_ignores_angles() -> None:
    model = LocalizeMeanPool(Encode())
    torch.manual_seed(0)
    views = torch.randn(1, 3, 1, 8, 8)
    out_a = model(views, angles=torch.zeros(1, 3))
    out_b = model(views, angles=torch.linspace(0, 2, 3).unsqueeze(0))
    assert torch.allclose(out_a, out_b)


def test_encode_views_shape() -> None:
    model = LocalizeMeanPool(Encode())
    views = torch.randn(2, 5, 1, 12, 12)
    emb = model.encode_views(views)
    assert emb.shape == (2, 5, 128)
