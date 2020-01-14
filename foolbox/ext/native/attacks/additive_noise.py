from abc import ABC
from abc import abstractmethod
import eagerpy as ep

from ..utils import flatten
from ..utils import atleast_kd


class BaseAdditiveNoiseAttack(ABC):
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, labels, *, epsilon):
        x = ep.astensor(inputs)
        min_, max_ = self.model.bounds()
        p = self.sample_noise(x)
        norms = self.get_norms(p)
        p = p / atleast_kd(norms, p.ndim)
        x = x + epsilon * p
        x = x.clip(min_, max_)
        return x.tensor

    @abstractmethod
    def sample_noise(self, x):
        raise NotImplementedError

    @abstractmethod
    def get_norms(self, p):
        raise NotImplementedError


class L2Mixin:
    def get_norms(self, p):
        return flatten(p).square().sum(axis=-1).sqrt()


class LinfMixin:
    def get_norms(self, p):
        return flatten(p).max(axis=-1)


class GaussianMixin:
    def sample_noise(self, x):
        return x.normal(x.shape)


class UniformMixin:
    def sample_noise(self, x):
        return x.uniform(x.shape, -1, 1)


class L2AdditiveGaussianNoiseAttack(L2Mixin, GaussianMixin, BaseAdditiveNoiseAttack):
    pass


class L2AdditiveUniformNoiseAttack(L2Mixin, UniformMixin, BaseAdditiveNoiseAttack):
    pass


class LinfAdditiveUniformNoiseAttack(LinfMixin, UniformMixin, BaseAdditiveNoiseAttack):
    pass


class BaseRepeatedAdditiveNoiseAttack(ABC):
    def __init__(self, model):
        self.model = model

    def __call__(
        self, inputs, labels, *, epsilon, criterion, repeats=100, check_trivial=True
    ):
        originals = ep.astensor(inputs)
        labels = ep.astensor(labels)

        def is_adversarial(p: ep.Tensor) -> ep.Tensor:
            """For each input in x, returns true if it is an adversarial for
            the given model and criterion"""
            logits = ep.astensor(self.model.forward(p.tensor))
            return criterion(originals, labels, p, logits)

        x0 = ep.astensor(inputs)
        min_, max_ = self.model.bounds()

        result = x0
        if check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x0, len(result)).bool()

        for _ in range(repeats):
            if found.all():
                break

            p = self.sample_noise(x0)
            norms = self.get_norms(p)
            p = p / atleast_kd(norms, p.ndim)
            x = x0 + epsilon * p
            x = x.clip(min_, max_)
            is_adv = is_adversarial(x)
            is_new_adv = ep.logical_and(is_adv, ep.logical_not(found))
            result = ep.where(atleast_kd(is_new_adv, x.ndim), x, result)
            found = ep.logical_or(found, is_adv)

        return result.tensor

    @abstractmethod
    def sample_noise(self, x):
        raise NotImplementedError

    @abstractmethod
    def get_norms(self, p):
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
