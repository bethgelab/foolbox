from functools import partial
from typing import Callable, Optional

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
        rel_stepsize: float = 0.2,
        abs_stepsize: Optional[float] = None,
        steps: int = 10,
        random_start: bool = False,
    ):
        self.momentum = momentum
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return GDMOptimizer(
            x, stepsize, self.momentum, partial(normalize_lp_norms, p=1)
        )


class L2MomentumIterativeFastGradientMethod(L2BasicIterativeAttack):
    def __init__(
        self,
        *,
        momentum: float = 1.0,
        rel_stepsize: float = 0.2,
        abs_stepsize: Optional[float] = None,
        steps: int = 10,
        random_start: bool = False,
    ):
        self.momentum = momentum
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return GDMOptimizer(
            x, stepsize, self.momentum, partial(normalize_lp_norms, p=2)
        )


class LinfMomentumIterativeFastGradientMethod(LinfBasicIterativeAttack):
    """Momentum Iterative Fast Gradient Sign Method (MI-FGSM) [#Dong18]

    References: .. [#Dong18] Dong Y, Liao F, Pang T, et al. Boosting adversarial attacks with momentum[
    C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 9185-9193.
    https://arxiv.org/abs/1710.06081
    """

    def __init__(
        self,
        *,
        momentum: float = 1.0,
        rel_stepsize: float = 0.2,
        abs_stepsize: Optional[float] = None,
        steps: int = 10,
        random_start: bool = False,
    ):
        self.momentum = momentum
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return GDMOptimizer(x, stepsize, self.momentum)
