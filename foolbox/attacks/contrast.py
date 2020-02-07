from typing import Union, Any
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..types import L2

from ..criteria import Criterion

from ..models import Model

from .base import FixedEpsilonAttack
from .base import T


class L2ContrastReductionAttack(FixedEpsilonAttack):
    """Reduces the contrast of the input using a perturbation of the given size

    Parameters
    ----------
    target
        Target relative to the bounds from 0 (min) to 1 (max) towards which the contrast is reduced
    """

    def __init__(self, epsilon: L2, target: float = 0.5) -> None:
        self.epsilon = epsilon
        self.target = target

    def __call__(
        self, model: Model, inputs: T, criterion: Union[Criterion, Any] = None
    ) -> T:
        x, restore_type = ep.astensor_(inputs)
        del inputs, criterion

        min_, max_ = model.bounds
        target = min_ + self.target * (max_ - min_)

        direction = target - x
        norms = ep.norms.l2(flatten(direction), axis=-1)
        scale = self.epsilon / atleast_kd(norms, direction.ndim)
        scale = ep.minimum(scale, 1)

        x = x + scale * direction
        x = x.clip(min_, max_)
        return restore_type(x)
