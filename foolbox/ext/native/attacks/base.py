from abc import ABC, abstractmethod
from typing import Callable, TypeVar
import eagerpy as ep

from ..models.base import Model
from ..criteria import Criterion


T = TypeVar("T")


class Attack(ABC):
    @abstractmethod
    def __call__(self, model: Model, inputs: T, labels: T) -> T:
        ...


class FixedEpsilonAttack(Attack):
    """Fixed-epsilon attacks try to find adversarials whose perturbation sizes are limited by a fixed epsilon"""

    pass


class MinimizationAttack(Attack):
    """Minimization attacks try to find adversarials with minimal perturbation sizes"""

    pass


# TODO: maybe automatically track all subclasses to make them user-queryable
# TODO: for minimization attacks, should the epsilon be part of __init__? (pro: stepsize, etc. depend on it; contra: initialize, then run with different epsilons)


def get_is_adversarial(
    criterion: Criterion, inputs: ep.Tensor, labels: ep.Tensor, model: Model
) -> Callable[[ep.Tensor], ep.Tensor]:
    def is_adversarial(perturbed: ep.Tensor) -> ep.Tensor:
        logits = model(perturbed)
        return criterion(inputs, labels, perturbed, logits)

    return is_adversarial
