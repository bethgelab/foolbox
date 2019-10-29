import numpy as np
import eagerpy as ep


def flatten(x: ep.Tensor) -> ep.Tensor:
    shape = (x.shape[0], -1)
    return x.reshape(shape)


def atleast_kd(x: ep.Tensor, k) -> ep.Tensor:
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)


def clip_l2_norms(x: ep.Tensor, norm) -> ep.Tensor:
    norms = flatten(x).square().sum(axis=-1).sqrt()
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = ep.minimum(1, norm / norms)  # clipping -> decreasing but not increasing
    factor = atleast_kd(factor, x.ndim)
    return x * factor


def normalize_l2_norms(x: ep.Tensor) -> ep.Tensor:
    norms = flatten(x).square().sum(axis=-1).sqrt()
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = 1 / norms
    factor = atleast_kd(factor, x.ndim)
    return x * factor


class L2BasicIterativeAttack:
    """L2 Basic Iterative Method"""

    def __init__(self, model):
        self.model = model

    def __call__(
        self, inputs, labels, *, rescale=True, epsilon=0.3, step_size=0.05, num_steps=10
    ):
        if rescale:
            min_, max_ = self.model.bounds()
            scale = (max_ - min_) * np.sqrt(np.prod(inputs.shape[1:]))
            epsilon = epsilon * scale
            step_size = step_size * scale

        x = ep.astensor(inputs)
        y = ep.astensor(labels)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 1

        x0 = x

        for _ in range(num_steps):
            gradients = ep.astensor(self.model.gradient(x.tensor, y.tensor))
            gradients = normalize_l2_norms(gradients)
            x = x + step_size * gradients
            x = x0 + clip_l2_norms(x - x0, epsilon)
            x = ep.clip(x, *self.model.bounds())

        return x.tensor


class LinfinityBasicIterativeAttack:
    """L-infinity Basic Iterative Method"""

    def __init__(self, model):
        self.model = model

    def __call__(
        self, inputs, labels, *, rescale=True, epsilon=0.3, step_size=0.05, num_steps=10
    ):
        if rescale:
            min_, max_ = self.model.bounds()
            scale = max_ - min_
            epsilon = epsilon * scale
            step_size = step_size * scale

        x = ep.astensor(inputs)
        y = ep.astensor(labels)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 1

        x0 = x

        for _ in range(num_steps):
            gradients = ep.astensor(self.model.gradient(x.tensor, y.tensor))
            gradients = gradients.sign()
            x = x + step_size * gradients
            x = x0 + ep.clip(x - x0, -epsilon, epsilon)
            x = ep.clip(x, *self.model.bounds())

        return x.tensor
