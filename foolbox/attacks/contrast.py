from typing import Union, Any
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..criteria import Criterion

from ..distances import l2

from ..models import Model

from .base import FixedEpsilonAttack
from .base import T
from .base import raise_if_kwargs


class L2ContrastReductionAttack(FixedEpsilonAttack):
    """Reduces the contrast of the input using a perturbation of the given size

    Args:
        target : Target relative to the bounds from 0 (min) to 1 (max)
            towards which the contrast is reduced
    """

    distance = l2

    def __init__(self, *, target: float = 0.5):
        self.target = target

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, criterion, kwargs

        min_, max_ = model.bounds
        target = min_ + self.target * (max_ - min_)

        direction = target - x
        norms = ep.norms.l2(flatten(direction), axis=-1)
        scale = epsilon / atleast_kd(norms, direction.ndim)
        scale = ep.minimum(scale, 1)

        x = x + scale * direction
        x = x.clip(min_, max_)
        return restore_type(x)
