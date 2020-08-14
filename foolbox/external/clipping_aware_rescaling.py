# Copyright (c) 2020, Jonas Rauber
#
# Licensed under the BSD 3-Clause License
#
# Last changed:
# * 2020-07-15
# * 2020-01-08
# * 2019-04-18

import eagerpy as ep


def l2_clipping_aware_rescaling(x, delta, eps: float, a: float = 0.0, b: float = 1.0):  # type: ignore
    """Calculates eta such that norm(clip(x + eta * delta, a, b) - x) == eps.

    Assumes x and delta have a batch dimension and eps, a, b, and p are
    scalars. If the equation cannot be solved because eps is too large, the
    left hand side is maximized.

    Args:
        x: A batch of inputs (PyTorch Tensor, TensorFlow Eager Tensor, NumPy
            Array, JAX Array, or EagerPy Tensor).
        delta: A batch of perturbation directions (same shape and type as x).
        eps: The target norm (non-negative float).
        a: The lower bound of the data domain (float).
        b: The upper bound of the data domain (float).

    Returns:
        eta: A batch of scales with the same number of dimensions as x but all
            axis == 1 except for the batch dimension.
    """
    (x, delta), restore_fn = ep.astensors_(x, delta)
    N = x.shape[0]
    assert delta.shape[0] == N
    rows = ep.arange(x, N)

    delta2 = delta.square().reshape((N, -1))
    space = ep.where(delta >= 0, b - x, x - a).reshape((N, -1))
    f2 = space.square() / ep.maximum(delta2, 1e-20)
    ks = ep.argsort(f2, axis=-1)
    f2_sorted = f2[rows[:, ep.newaxis], ks]
    m = ep.cumsum(delta2[rows[:, ep.newaxis], ks.flip(axis=1)], axis=-1).flip(axis=1)
    dx = f2_sorted[:, 1:] - f2_sorted[:, :-1]
    dx = ep.concatenate((f2_sorted[:, :1], dx), axis=-1)
    dy = m * dx
    y = ep.cumsum(dy, axis=-1)
    c = y >= eps ** 2

    # work-around to get first nonzero element in each row
    f = ep.arange(x, c.shape[-1], 0, -1)
    j = ep.argmax(c.astype(f.dtype) * f, axis=-1)

    eta2 = f2_sorted[rows, j] - (y[rows, j] - eps ** 2) / m[rows, j]
    # it can happen that for certain rows even the largest j is not large enough
    # (i.e. c[:, -1] is False), then we will just use it (without any correction) as it's
    # the best we can do (this should also be the only cases where m[j] can be
    # 0 and they are thus not a problem)
    eta2 = ep.where(c[:, -1], eta2, f2_sorted[:, -1])
    eta = ep.sqrt(eta2)
    eta = eta.reshape((-1,) + (1,) * (x.ndim - 1))

    # xp = ep.clip(x + eta * delta, a, b)
    # l2 = (xp - x).reshape((N, -1)).square().sum(axis=-1).sqrt()
    return restore_fn(eta)
