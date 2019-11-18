import numpy as np
import eagerpy as ep

from ..utils import flatten
from ..utils import atleast_kd


class L2ContrastReductionAttack:
    """Reduces the contrast of the input using a perturbation of the given l2 norm"""

    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, labels, *, l2):
        x = ep.astensor(inputs)
        min_, max_ = self.model.bounds()
        target = (max_ + min_) / 2
        v = target - x
        norms = flatten(v).square().sum(axis=-1).sqrt()
        x = x + l2 / atleast_kd(norms, v.ndim) * v
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

        lower_bound = np.zeros((N,))
        upper_bound = np.ones((N,))

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

    def __call__(self, inputs, labels, *, num_steps=1000):
        x = ep.astensor(inputs)
        y = ep.astensor(labels)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 1

        min_, max_ = self.model.bounds()
        target = (max_ + min_) / 2
        v = target - x

        N = len(x)
        x0 = x

        epsilons = np.linspace(0, 1, num=num_steps + 1)
        best = np.ones((N,))

        for epsilon in epsilons:
            # TODO: reduce the batch size to the ones that haven't been sucessful
            x = x0 + epsilon * v
            logits = ep.astensor(self.model.forward(x.tensor))
            classes = logits.argmax(axis=-1)
            is_adv = (classes != labels).numpy()

            best = np.minimum(
                np.logical_not(is_adv).astype(np.float32)
                + is_adv.astype(np.float32) * epsilon,
                best,
            )

        best = ep.from_numpy(x0, best)
        x = x0 + atleast_kd(best, x0.ndim) * v
        return x.tensor
