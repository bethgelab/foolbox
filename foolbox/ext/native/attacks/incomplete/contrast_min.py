import numpy as np
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..types import L2

from ..models import Model

from ..criteria import Criterion
from ..criteria import misclassification

from .base import FixedEpsilonAttack
from .base import MinimizationAttack
from .base import T
from .base import get_is_adversarial


class BinarySearchContrastReductionAttack(MinimizationAttack):
    """Reduces the contrast of the input using a binary search to find the
    smallest adversarial perturbation"""

    def __init__(self, binary_search_steps: int = 15) -> None:
        self.binary_search_steps = binary_search_steps

    def __call__(
        self,
        model: Model,
        inputs: T,
        labels: T,
        *,
        criterion: Criterion = misclassification,
    ) -> T:
        x, y, restore = ep.astensors_(inputs, labels)
        del inputs, labels

        is_adversarial = get_is_adversarial(criterion, x, y, model)

        x = inputs
        min_, max_ = model.bounds()
        target = (max_ + min_) / 2
        v = target - x

        N = len(x)
        x0 = x

        npdtype = x.numpy().dtype
        lower_bound = np.zeros((N,), dtype=npdtype)
        upper_bound = np.ones((N,), dtype=npdtype)

        epsilons = lower_bound
        for _ in range(self.binary_search_steps):
            x = x0 + atleast_kd(ep.from_numpy(x0, epsilons), x0.ndim) * v
            is_adv = is_adversarial(x)
            lower_bound = np.where(is_adv, lower_bound, epsilons)
            upper_bound = np.where(is_adv, epsilons, upper_bound)
            epsilons = (lower_bound + upper_bound) / 2

        epsilons = upper_bound
        epsilons = ep.from_numpy(x0, epsilons)
        x = x0 + atleast_kd(epsilons, x0.ndim) * v
        return restore(x)


class LinearSearchContrastReductionAttack:
    """Reduces the contrast of the input using a linear search to find the smallest adversarial perturbation"""

    def __init__(self, steps=1000):
        self.steps = steps

    def __call__(self, model, inputs, labels, *, criterion=misclassification):
        inputs, labels, restore = wrap(inputs, labels)
        is_adversarial = get_is_adversarial(criterion, inputs, labels, model)

        x = inputs
        min_, max_ = self.model.bounds()
        target = (max_ + min_) / 2
        v = target - x

        N = len(x)
        x0 = x

        npdtype = x.numpy().dtype
        epsilons = np.linspace(0, 1, num=self.steps + 1, dtype=npdtype)
        best = np.ones((N,), dtype=npdtype)

        for epsilon in epsilons:
            # TODO: reduce the batch size to the ones that haven't been sucessful
            x = x0 + epsilon * v
            is_adv = is_adversarial(x).numpy()

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
