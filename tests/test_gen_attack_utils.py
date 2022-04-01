import eagerpy as ep
import numpy as np
import pytest
from typing import Any

from foolbox.attacks.gen_attack_utils import rescale_images


def test_rescale_axis(request: Any, dummy: ep.Tensor) -> None:
    backend = request.config.option.backend
    if backend == "numpy":
        pytest.skip()

    x_np = np.random.uniform(0.0, 1.0, size=(16, 3, 64, 64))
    x_np_ep = ep.astensor(x_np)
    x_up_np_ep = rescale_images(x_np_ep, (16, 3, 128, 128), 1)
    x_up_np = x_up_np_ep.numpy()

    x = ep.from_numpy(dummy, x_np)
    x_ep = ep.astensor(x)
    x_up_ep = rescale_images(x_ep, (16, 3, 128, 128), 1)
    x_up = x_up_ep.numpy()

    assert np.allclose(x_up_np, x_up, atol=1e-5)


def test_rescale_axis_nhwc(request: Any, dummy: ep.Tensor) -> None:
    backend = request.config.option.backend
    if backend == "numpy":
        pytest.skip()

    x_np = np.random.uniform(0.0, 1.0, size=(16, 64, 64, 3))
    x_np_ep = ep.astensor(x_np)
    x_up_np_ep = rescale_images(x_np_ep, (16, 128, 128, 3), -1)
    x_up_np = x_up_np_ep.numpy()

    x = ep.from_numpy(dummy, x_np)
    x_ep = ep.astensor(x)
    x_up_ep = rescale_images(x_ep, (16, 128, 128, 3), -1)
    x_up = x_up_ep.numpy()

    assert np.allclose(x_up_np, x_up, atol=1e-5)
