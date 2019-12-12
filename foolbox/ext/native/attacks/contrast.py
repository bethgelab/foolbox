import numpy as np
import eagerpy as ep

from ..utils import flatten
from ..utils import atleast_kd


class L2ContrastReductionAttack:
    """Reduces the contrast of the input using a perturbation of the given l2 norm"""

    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, labels, *, epsilon):
        x = ep.astensor(inputs)
        min_, max_ = self.model.bounds()
        target = (max_ + min_) / 2
        v = target - x
        norms = flatten(v).square().sum(axis=-1).sqrt()
        scale = epsilon / atleast_kd(norms, v.ndim)
        scale = ep.minimum(scale, 1)
        x = x + scale * v
        x = x.clip(min_, max_)
        return x.tensor


class BinarySearchContrastReductionAttack:
    """Reduces the contrast of the input using a binary search to find the smallest adversarial perturbation"""

    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, labels, *, binary_search_steps=15):
        x = ep.astensor(inputs)
        y = ep.astensor(labels)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 1

        min_, max_ = self.model.bounds()
        target = (max_ + min_) / 2
        v = target - x

        N = len(x)
        x0 = x

        npdtype = x.numpy().dtype
        lower_bound = np.zeros((N,), dtype=npdtype)
        upper_bound = np.ones((N,), dtype=npdtype)

        epsilons = lower_bound

        for _ in range(binary_search_steps):
            x = x0 + atleast_kd(ep.from_numpy(x0, epsilons), x0.ndim) * v
            logits = ep.astensor(self.model.forward(x.tensor))
            classes = logits.argmax(axis=-1)
            is_adv = (classes != labels).numpy()
            lower_bound = np.where(is_adv, lower_bound, epsilons)
            upper_bound = np.where(is_adv, epsilons, upper_bound)
            epsilons = (lower_bound + upper_bound) / 2

        epsilons = upper_bound
        epsilons = ep.from_numpy(x0, epsilons)
        x = x0 + atleast_kd(epsilons, x0.ndim) * v
        return x.tensor


class LinearSearchContrastReductionAttack:
    """Reduces the contrast of the input using a linear search to find the smallest adversarial perturbation"""

    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, labels, *, steps=1000):
        x = ep.astensor(inputs)
        y = ep.astensor(labels)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 1

        min_, max_ = self.model.bounds()
        target = (max_ + min_) / 2
        v = target - x

        N = len(x)
        x0 = x

        npdtype = x.numpy().dtype
        epsilons = np.linspace(0, 1, num=steps + 1, dtype=npdtype)
        best = np.ones((N,), dtype=npdtype)

        for epsilon in epsilons:
            # TODO: reduce the batch size to the ones that haven't been sucessful
            x = x0 + epsilon * v
            logits = ep.astensor(self.model.forward(x.tensor))
            classes = logits.argmax(axis=-1)
            is_adv = (classes != labels).numpy()

            best = np.minimum(
                np.logical_not(is_adv).astype(npdtype)
                + is_adv.astype(npdtype) * epsilon,
                best,
            )

            if (best < 1).all():
                break

        best = ep.from_numpy(x0, best)
        x = x0 + atleast_kd(best, x0.ndim) * v
        return x.tensor
