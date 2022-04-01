from typing import Union, Optional, Any
import numpy as np
import eagerpy as ep

from ..devutils import atleast_kd

from ..distances import Distance

from .base import FlexibleDistanceMinimizationAttack
from .base import Model
from .base import Criterion
from .base import T
from .base import get_is_adversarial
from .base import get_criterion
from .base import raise_if_kwargs

import warnings


class LinearSearchBlendedUniformNoiseAttack(FlexibleDistanceMinimizationAttack):
    """Blends the input with a uniform noise input until it is misclassified.

    Args:
        distance : Distance measure for which minimal adversarial examples are searched.
        directions : Number of random directions in which the perturbation is searched.
        steps : Number of blending steps between the original image and the random
            directions.
    """

    def __init__(
        self,
        *,
        distance: Optional[Distance] = None,
        directions: int = 1000,
        steps: int = 1000,
    ):
        super().__init__(distance=distance)
        self.directions = directions
        self.steps = steps

        if directions <= 0:
            raise ValueError("directions must be larger than 0")

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        is_adversarial = get_is_adversarial(criterion_, model)

        min_, max_ = model.bounds

        N = len(x)

        for j in range(self.directions):
            # random noise inputs tend to be classified into the same class,
            # so we might need to make very many draws if the original class
            # is that one
            random_ = ep.uniform(x, x.shape, min_, max_)
            is_adv_ = atleast_kd(is_adversarial(random_), x.ndim)

            if j == 0:
                random = random_
                is_adv = is_adv_
            else:
                random = ep.where(is_adv, random, random_)
                is_adv = is_adv.logical_or(is_adv_)

            if is_adv.all():
                break

        if not is_adv.all():
            warnings.warn(
                f"{self.__class__.__name__} failed to draw sufficient random"
                f" inputs that are adversarial ({is_adv.sum()} / {N})."
            )

        x0 = x

        epsilons = np.linspace(0, 1, num=self.steps + 1, dtype=np.float32)
        best = ep.ones(x, (N,))

        for epsilon in epsilons:
            x = (1 - epsilon) * x0 + epsilon * random
            # TODO: due to limited floating point precision, clipping can be required
            is_adv = is_adversarial(x)

            epsilon = epsilon.item()

            best = ep.minimum(ep.where(is_adv, epsilon, 1.0), best)

            if (best < 1).all():
                break

        best = atleast_kd(best, x0.ndim)
        x = (1 - best) * x0 + best * random

        return restore_type(x)
