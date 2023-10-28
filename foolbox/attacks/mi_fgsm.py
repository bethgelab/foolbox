from functools import partial
from typing import Callable

from foolbox.attacks.gradient_descent_base import normalize_lp_norms

from .basic_iterative_method import (
    Optimizer,
    L1BasicIterativeAttack,
    L2BasicIterativeAttack,
    LinfBasicIterativeAttack,
)
import eagerpy as ep


class GDMOptimizer(Optimizer):
    # create GD optimizer with momentum
    def __init__(
        self,
        x: ep.Tensor,
        stepsize: float,
        momentum: float = 1.0,
        normalize_fn: Callable[[ep.Tensor], ep.Tensor] = lambda x: x.sign(),
    ):
        self.stepsize = stepsize
        self.momentum = momentum
        self.normalize = normalize_fn
        self.accumulation_grad = ep.zeros_like(x)

    def __call__(self, gradient: ep.Tensor) -> ep.Tensor:
        self.accumulation_grad = self.momentum * self.accumulation_grad + gradient
        return self.stepsize * self.normalize(self.accumulation_grad)


class L1MomentumIterativeFastGradientMethod(L1BasicIterativeAttack):
    def __init__(
        self,
        *,
        momentum: float = 1.0,
        **kwargs,
    ):
        self.momentum = momentum
        super().__init__(**kwargs)

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return GDMOptimizer(
            x, stepsize, self.momentum, partial(normalize_lp_norms, p=1)
        )


class L2MomentumIterativeFastGradientMethod(L2BasicIterativeAttack):
    def __init__(
        self,
        *,
        momentum: float = 1.0,
        **kwargs,
    ):
        self.momentum = momentum
        super().__init__(**kwargs)

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return GDMOptimizer(
            x, stepsize, self.momentum, partial(normalize_lp_norms, p=2)
        )


class LinfMomentumIterativeFastGradientMethod(LinfBasicIterativeAttack):
    """create I-FGSM with Momentum [#Dong18]

    References: .. [#Dong18] Dong Y, Liao F, Pang T, et al. Boosting adversarial attacks with momentum[
    C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 9185-9193.
    https://arxiv.org/abs/1607.02533
    """

    def __init__(
        self,
        *,
        momentum: float = 1.0,
        **kwargs,
    ):
        self.momentum = momentum
        super().__init__(**kwargs)

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return GDMOptimizer(x, stepsize, self.momentum)
