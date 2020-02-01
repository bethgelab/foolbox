from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Union
import eagerpy as ep

from ..models.base import Model
from ..criteria import Criterion
from ..criteria import Misclassification


T = TypeVar("T")


class Attack(ABC):
    @abstractmethod
    def __call__(self, model: Model, inputs: T, criterion_or_labels) -> T:
        # in principle, the type of criterion_or_labels is Union[Criterion, T]
        # but we want to give subclasses the option to specify the supported
        # criteria explicitly (i.e. specifying a stricter type constraint)
        ...


class FixedEpsilonAttack(Attack):
    """Fixed-epsilon attacks try to find adversarials whose perturbation sizes
    are limited by a fixed epsilon"""

    pass


class MinimizationAttack(Attack):
    """Minimization attacks try to find adversarials with minimal perturbation sizes"""

    pass


def get_is_adversarial(
    criterion: Criterion, model: Model
) -> Callable[[ep.Tensor], ep.Tensor]:
    def is_adversarial(perturbed: ep.Tensor) -> ep.Tensor:
        outputs = model(perturbed)
        return criterion(perturbed, outputs)

    return is_adversarial


def get_criterion(criterion_or_labels: Union[Criterion, T]) -> Criterion:
    if isinstance(criterion_or_labels, Criterion):
        criterion = criterion_or_labels
    else:
        criterion = Misclassification(criterion_or_labels)
    return criterion
