from typing import Union, Any, Optional
import eagerpy as ep

from ..devutils import atleast_kd

from ..models import Model

from ..criteria import Criterion

from ..distances import Distance

from .base import FlexibleDistanceMinimizationAttack
from .base import T
from .base import get_is_adversarial
from .base import get_criterion
from .base import raise_if_kwargs


class BinarySearchContrastReductionAttack(FlexibleDistanceMinimizationAttack):
    """Reduces the contrast of the input using a binary search to find the
    smallest adversarial perturbation

    Args:
        distance : Distance measure for which minimal adversarial examples are searched.
        binary_search_steps : Number of iterations in the binary search.
            This controls the precision of the results.
        target : Target relative to the bounds from 0 (min) to 1 (max)
            towards which the contrast is reduced
    """

    def __init__(
        self,
        *,
        distance: Optional[Distance] = None,
        binary_search_steps: int = 15,
        target: float = 0.5,
    ):
        super().__init__(distance=distance)
        self.binary_search_steps = binary_search_steps
        self.target = target

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        min_, max_ = model.bounds
        target = min_ + self.target * (max_ - min_)
        direction = target - x

        lower_bound = ep.zeros(x, len(x))
        upper_bound = ep.ones(x, len(x))
        epsilons = lower_bound
        for _ in range(self.binary_search_steps):
            eps = atleast_kd(epsilons, x.ndim)
            is_adv = is_adversarial(x + eps * direction)
            lower_bound = ep.where(is_adv, lower_bound, epsilons)
            upper_bound = ep.where(is_adv, epsilons, upper_bound)
            epsilons = (lower_bound + upper_bound) / 2

        epsilons = upper_bound
        eps = atleast_kd(epsilons, x.ndim)
        xp = x + eps * direction
        return restore_type(xp)


class LinearSearchContrastReductionAttack(FlexibleDistanceMinimizationAttack):
    """Reduces the contrast of the input using a linear search to find the
    smallest adversarial perturbation"""

    def __init__(
        self,
        *,
        distance: Optional[Distance] = None,
        steps: int = 1000,
        target: float = 0.5,
    ):
        super().__init__(distance=distance)
        self.steps = steps
        self.target = target

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        min_, max_ = model.bounds
        target = min_ + self.target * (max_ - min_)
        direction = target - x

        best = ep.ones(x, len(x))

        epsilon = 0.0
        stepsize = 1.0 / self.steps
        for _ in range(self.steps):
            # TODO: reduce the batch size to the ones that have not yet been sucessful

            is_adv = is_adversarial(x + epsilon * direction)
            is_best_adv = ep.logical_and(is_adv, best == 1)
            best = ep.where(is_best_adv, epsilon, best)

            if (best < 1).all():
                break  # pragma: no cover

            epsilon += stepsize

        eps = atleast_kd(best, x.ndim)
        xp = x + eps * direction
        return restore_type(xp)
