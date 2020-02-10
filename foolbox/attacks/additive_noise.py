from typing import Union
from abc import ABC
from abc import abstractmethod
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from .base import FixedEpsilonAttack
from .base import Criterion
from .base import Model
from .base import T, Any
from .base import get_criterion
from .base import get_is_adversarial


class BaseAdditiveNoiseAttack(FixedEpsilonAttack, ABC):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(
        self, model: Model, inputs: T, criterion: Union[Criterion, Any] = None
    ) -> T:
        x, restore_type = ep.astensor_(inputs)
        del inputs, criterion

        min_, max_ = model.bounds
        p = self.sample_noise(x)
        norms = self.get_norms(p)
        p = p / atleast_kd(norms, p.ndim)
        x = x + self.epsilon * p
        x = x.clip(min_, max_)

        return restore_type(x)

    @abstractmethod
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_norms(self, p: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError


class L2Mixin:
    def get_norms(self, p: ep.Tensor) -> ep.Tensor:
        return flatten(p).square().sum(axis=-1).sqrt()


class LinfMixin:
    def get_norms(self, p: ep.Tensor) -> ep.Tensor:
        return flatten(p).max(axis=-1)


class GaussianMixin:
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        return x.normal(x.shape)


class UniformMixin:
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        return x.uniform(x.shape, -1, 1)


class L2AdditiveGaussianNoiseAttack(L2Mixin, GaussianMixin, BaseAdditiveNoiseAttack):
    pass


class L2AdditiveUniformNoiseAttack(L2Mixin, UniformMixin, BaseAdditiveNoiseAttack):
    pass


class LinfAdditiveUniformNoiseAttack(LinfMixin, UniformMixin, BaseAdditiveNoiseAttack):
    pass


class BaseRepeatedAdditiveNoiseAttack(FixedEpsilonAttack, ABC):
    def __init__(self, epsilon: float, repeats: int = 100, check_trivial: bool = True):
        self.epsilon = epsilon
        self.repeats = repeats
        self.check_trivial = check_trivial

    def __call__(
        self, model: Model, inputs: T, criterion: Union[Criterion, Any] = None
    ) -> T:
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion

        is_adversarial = get_is_adversarial(criterion_, model)

        min_, max_ = model.bounds

        result = x0
        if self.check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x0, len(result)).bool()

        for _ in range(self.repeats):
            if found.all():
                break

            p = self.sample_noise(x0)
            norms = self.get_norms(p)
            p = p / atleast_kd(norms, p.ndim)
            x = x0 + self.epsilon * p
            x = x.clip(min_, max_)
            is_adv = is_adversarial(x)
            is_new_adv = ep.logical_and(is_adv, ep.logical_not(found))
            result = ep.where(atleast_kd(is_new_adv, x.ndim), x, result)
            found = ep.logical_or(found, is_adv)

        return restore_type(result)

    @abstractmethod
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_norms(self, p: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError


class L2RepeatedAdditiveGaussianNoiseAttack(
    L2Mixin, GaussianMixin, BaseRepeatedAdditiveNoiseAttack
):
    pass


class L2RepeatedAdditiveUniformNoiseAttack(
    L2Mixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack
):
    pass


class LinfRepeatedAdditiveUniformNoiseAttack(
    LinfMixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack
):
    pass
