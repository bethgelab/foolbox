from typing import Union, Any, cast
from abc import ABC
from abc import abstractmethod
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..distances import l2, linf

from .base import FixedEpsilonAttack
from .base import Criterion
from .base import Model
from .base import T
from .base import get_criterion
from .base import get_is_adversarial
from .base import raise_if_kwargs

from ..external.clipping_aware_rescaling import l2_clipping_aware_rescaling


class BaseAdditiveNoiseAttack(FixedEpsilonAttack, ABC):
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
        p = self.sample_noise(x)
        epsilons = self.get_epsilons(x, p, epsilon, min_=min_, max_=max_)
        x = x + epsilons * p
        x = x.clip(min_, max_)

        return restore_type(x)

    @abstractmethod
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        raise NotImplementedError


class L2Mixin:
    distance = l2

    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        norms = flatten(p).norms.l2(axis=-1)
        return epsilon / atleast_kd(norms, p.ndim)


class L2ClippingAwareMixin:
    distance = l2

    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        return cast(
            ep.Tensor, l2_clipping_aware_rescaling(x, p, epsilon, a=min_, b=max_)
        )


class LinfMixin:
    distance = linf

    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        norms = flatten(p).max(axis=-1)
        return epsilon / atleast_kd(norms, p.ndim)


class GaussianMixin:
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        return x.normal(x.shape)


class UniformMixin:
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        return x.uniform(x.shape, -1, 1)


class L2AdditiveGaussianNoiseAttack(L2Mixin, GaussianMixin, BaseAdditiveNoiseAttack):
    """Samples Gaussian noise with a fixed L2 size."""

    pass


class L2AdditiveUniformNoiseAttack(L2Mixin, UniformMixin, BaseAdditiveNoiseAttack):
    """Samples uniform noise with a fixed L2 size."""

    pass


class L2ClippingAwareAdditiveGaussianNoiseAttack(
    L2ClippingAwareMixin, GaussianMixin, BaseAdditiveNoiseAttack
):
    """Samples Gaussian noise with a fixed L2 size after clipping.

    The implementation is based on [#Rauber20]_.

    References:
        .. [#Rauber20] Jonas Rauber, Matthias Bethge
            "Fast Differentiable Clipping-Aware Normalization and Rescaling"
            https://arxiv.org/abs/2007.07677

    """

    pass


class L2ClippingAwareAdditiveUniformNoiseAttack(
    L2ClippingAwareMixin, UniformMixin, BaseAdditiveNoiseAttack
):
    """Samples uniform noise with a fixed L2 size after clipping.

    The implementation is based on [#Rauber20]_.

    References:
        .. [#Rauber20] Jonas Rauber, Matthias Bethge
            "Fast Differentiable Clipping-Aware Normalization and Rescaling"
            https://arxiv.org/abs/2007.07677

    """

    pass


class LinfAdditiveUniformNoiseAttack(LinfMixin, UniformMixin, BaseAdditiveNoiseAttack):
    """Samples uniform noise with a fixed L-infinity size"""

    pass


class BaseRepeatedAdditiveNoiseAttack(FixedEpsilonAttack, ABC):
    def __init__(self, *, repeats: int = 100, check_trivial: bool = True):
        self.repeats = repeats
        self.check_trivial = check_trivial

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
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

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
            epsilons = self.get_epsilons(x0, p, epsilon, min_=min_, max_=max_)
            x = x0 + epsilons * p
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
    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        raise NotImplementedError


class L2RepeatedAdditiveGaussianNoiseAttack(
    L2Mixin, GaussianMixin, BaseRepeatedAdditiveNoiseAttack
):
    """Repeatedly samples Gaussian noise with a fixed L2 size.

    Args:
        repeats : How often to sample random noise.
        check_trivial : Check whether original sample is already adversarial.
    """

    pass


class L2RepeatedAdditiveUniformNoiseAttack(
    L2Mixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack
):
    """Repeatedly samples uniform noise with a fixed L2 size.

    Args:
        repeats : How often to sample random noise.
        check_trivial : Check whether original sample is already adversarial.
    """

    pass


class L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(
    L2ClippingAwareMixin, GaussianMixin, BaseRepeatedAdditiveNoiseAttack
):
    """Repeatedly samples Gaussian noise with a fixed L2 size after clipping.

    The implementation is based on [#Rauber20]_.

    References:
        .. [#Rauber20] Jonas Rauber, Matthias Bethge
            "Fast Differentiable Clipping-Aware Normalization and Rescaling"
            https://arxiv.org/abs/2007.07677

    Args:
        repeats : How often to sample random noise.
        check_trivial : Check whether original sample is already adversarial.
    """

    pass


class L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(
    L2ClippingAwareMixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack
):
    """Repeatedly samples uniform noise with a fixed L2 size after clipping.

    The implementation is based on [#Rauber20]_.

    References:
        .. [#Rauber20] Jonas Rauber, Matthias Bethge
            "Fast Differentiable Clipping-Aware Normalization and Rescaling"
            https://arxiv.org/abs/2007.07677

    Args:
        repeats : How often to sample random noise.
        check_trivial : Check whether original sample is already adversarial.
    """

    pass


class LinfRepeatedAdditiveUniformNoiseAttack(
    LinfMixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack
):
    """Repeatedly samples uniform noise with a fixed L-infinity size.

    Args:
        repeats : How often to sample random noise.
        check_trivial : Check whether original sample is already adversarial.
    """

    pass
