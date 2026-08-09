"""WIN 3A.1 Fourier-coded spatial pooling smoke tests."""

from __future__ import annotations

import torch

from tomography_ml.localization import (
    Encode,
    FourierCodedPool2d,
    LocalizeSingleViewFourier,
    count_parameters,
    enumerate_fourier_modes,
    make_fourier,
    materialize_lazy_modules,
)


def test_enumerate_fourier_modes_starts_with_const() -> None:
    modes = enumerate_fourier_modes(5)
    assert modes[0] == (0, 0, "const")
    assert len(modes) == 5
    assert modes[1] == (1, 0, "cos")
    assert modes[2] == (1, 0, "sin")


def test_fourier_pool_output_shape_and_no_grad_on_basis() -> None:
    pool = FourierCodedPool2d(4)
    x = torch.randn(2, 4, 8, 10, requires_grad=True)
    out = pool(x)
    assert out.shape == (2, 4)
    assert pool._basis.shape == (4, 8, 10)
    assert not any(p.requires_grad for p in pool.buffers())
    out.sum().backward()
    assert x.grad is not None


def test_localize_single_view_fourier_forward_backward() -> None:
    model = LocalizeSingleViewFourier(Encode(), n_outputs=3, head_type="linear")
    x = torch.randn(3, 1, 16, 16, requires_grad=True)
    out = model(x)
    assert out.shape == (3, 3)
    assert len(model.mode_list) == Encode().out_channels
    n_params = count_parameters(model)
    # Learned params should stay far below a full Flatten head.
    assert n_params < 100_000
    out.sum().backward()
    assert x.grad is not None


def test_make_fourier_builder() -> None:
    model = make_fourier(n_outputs=2, downsample="base", head_type="linear")
    materialize_lazy_modules(model, torch.zeros(1, 1, 12, 12))
    assert model(torch.randn(2, 1, 12, 12)).shape == (2, 2)
