from typing import Callable, TypeVar, Any, Union
from abc import ABC, abstractmethod
import eagerpy as ep

from ..models.base import Model

from ..criteria import Criterion
from ..criteria import Misclassification

from ..devutils import atleast_kd


T = TypeVar("T")
CriterionType = TypeVar("CriterionType", bound=Criterion)


class Attack(ABC):
    @abstractmethod
    def __call__(self, model: Model, inputs: T, criterion: Any) -> T:
        # in principle, the type of criterion is Union[Criterion, T]
        # but we want to give subclasses the option to specify the supported
        # criteria explicitly (i.e. specifying a stricter type constraint)
        ...

    @abstractmethod
    def repeat(self, times: int) -> "Attack":
        ...


class AttackWrapper(Attack):
    pass


class Repeated(AttackWrapper):
    """Repeats the wrapped attack and returns the best result"""

    def __init__(self, attack: Attack, times: int):
        if not isinstance(attack, FixedEpsilonAttack):
            raise NotImplementedError  # TODO

        if times < 1:
            raise ValueError(f"expected times >= 1, got {times}")

        self._attack = attack
        self._times = times

    def __call__(self, model: Model, inputs: T, criterion: Union[Criterion, T]) -> T:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        best = self._attack(model, x, criterion)
        best_is_adv = is_adversarial(best)

        for _ in range(1, self._times):
            xp = self._attack(model, x, criterion)
            # assumes xp does not violate the perturbation size constraint

            is_adv = is_adversarial(xp)
            new_best = ep.logical_and(is_adv, best_is_adv.logical_not())

            best = ep.where(atleast_kd(new_best, best.ndim), xp, best)
            best_is_adv = ep.logical_or(is_adv, best_is_adv)

        return restore_type(best)

    def repeat(self, times: int) -> "Repeated":
        return Repeated(self._attack, self._times * times)


class FixedEpsilonAttack(Attack):
    """Fixed-epsilon attacks try to find adversarials whose perturbation sizes
    are limited by a fixed epsilon"""

    def repeat(self, times: int) -> Attack:
        return Repeated(self, times)


class MinimizationAttack(Attack):
    """Minimization attacks try to find adversarials with minimal perturbation sizes"""

    def repeat(self, times: int) -> Attack:
        raise NotImplementedError  # TODO


def get_is_adversarial(
    criterion: Criterion, model: Model
) -> Callable[[ep.Tensor], ep.Tensor]:
    def is_adversarial(perturbed: ep.Tensor) -> ep.Tensor:
        outputs = model(perturbed)
        return criterion(perturbed, outputs)

    return is_adversarial


def get_criterion(criterion: Union[Criterion, Any]) -> Criterion:
    if isinstance(criterion, Criterion):
        return criterion
    else:
        return Misclassification(criterion)
