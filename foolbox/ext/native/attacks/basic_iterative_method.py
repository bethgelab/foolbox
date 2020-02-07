from typing import Union
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..types import L2, Linf

from ..models.base import Model

from ..criteria import Misclassification

from .base import FixedEpsilonAttack
from .base import T
from .base import get_criterion


def clip_l2_norms(x: ep.Tensor, norm: float) -> ep.Tensor:
    norms = flatten(x).square().sum(axis=-1).sqrt()
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = ep.minimum(1, norm / norms)  # clipping -> decreasing but not increasing
    factor = atleast_kd(factor, x.ndim)
    return x * factor


def normalize_l2_norms(x: ep.Tensor) -> ep.Tensor:
    norms = flatten(x).square().sum(axis=-1).sqrt()
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = 1 / norms
    factor = atleast_kd(factor, x.ndim)
    return x * factor


class L2BasicIterativeAttack(FixedEpsilonAttack):
    """L2 Basic Iterative Method"""

    # TODO: document the typical hyperparameters for inputs in [0, 1]
    # epsilon = 2.0
    # stepsize = 0.4

    def __init__(self, epsilon: L2, stepsize: float, steps: int = 10):
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.steps = steps

    def __call__(
        self, model: Model, inputs: T, criterion: Union[Misclassification, T]
    ) -> T:
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion

        if not isinstance(criterion_, Misclassification):
            raise ValueError("unsupported criterion")

        labels = criterion_.labels

        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return ep.crossentropy(logits, labels).sum()

        x = x0
        for _ in range(self.steps):
            _, gradients = ep.value_and_grad(loss_fn, x)
            gradients = normalize_l2_norms(gradients)
            x = x + self.stepsize * gradients
            x = x0 + clip_l2_norms(x - x0, self.epsilon)
            x = ep.clip(x, *model.bounds)

        return restore_type(x)


class LinfBasicIterativeAttack(FixedEpsilonAttack):
    """L-infinity Basic Iterative Method"""

    def __init__(
        self,
        epsilon: Linf,
        stepsize: float,
        steps: int = 10,
        random_start: bool = False,
    ):
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.steps = steps
        self.random_start = random_start

        # TODO: document typical parameters for [0, 1]
        # epsilon=0.3,
        # step_size=0.05,

    def __call__(
        self, model: Model, inputs: T, criterion: Union[Misclassification, T]
    ) -> T:
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion

        if not isinstance(criterion_, Misclassification):
            raise ValueError("unsupported criterion")

        labels = criterion_.labels

        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return ep.crossentropy(logits, labels).sum()

        x = x0

        if self.random_start:
            x = x + ep.uniform(x, x.shape, -self.epsilon, self.epsilon)
            x = ep.clip(x, *model.bounds)

        for _ in range(self.steps):
            _, gradients = ep.value_and_grad(loss_fn, x)
            gradients = gradients.sign()
            x = x + self.stepsize * gradients
            x = x0 + ep.clip(x - x0, -self.epsilon, self.epsilon)
            x = ep.clip(x, *model.bounds)

        return restore_type(x)
