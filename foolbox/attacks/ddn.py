from typing import Union, Tuple
import eagerpy as ep

from ..models import Model
import math
from ..criteria import Misclassification, TargetedMisclassification

from ..devutils import atleast_kd, flatten

from .base import FixedEpsilonAttack
from .base import get_criterion
from .base import T


def normalize_l2_norms(x: ep.Tensor) -> ep.Tensor:
    norms = flatten(x).square().sum(axis=-1).sqrt()
    norms = ep.maximum(norms, 1e-12)  # avoid division by zero
    factor = 1 / norms
    factor = atleast_kd(factor, x.ndim)
    return x * factor


class DDNAttack(FixedEpsilonAttack):
    """DDN Attack"""

    def __init__(
        self,
        rescale: bool = False,
        epsilon: float = 2.0,
        init_epsilon: float = 1.0,
        steps: int = 10,
        gamma: float = 0.05,
    ):

        self.rescale = rescale
        self.epsilon = epsilon
        self.init_epsilon = init_epsilon
        self.steps = steps
        self.gamma = gamma

    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
    ) -> T:

        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion

        N = len(x)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
        else:
            raise ValueError("unsupported criterion")

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        if self.rescale:
            min_, max_ = model.bounds
            scale = (max_ - min_) * math.sqrt(flatten(x).shape[-1])
            init_epsilon = self.epsilon * scale
        else:
            init_epsilon = self.epsilon

        stepsize = ep.ones(x, len(x))

        def loss_fn(
            inputs: ep.Tensor, labels: ep.Tensor
        ) -> Tuple[ep.Tensor, ep.Tensor]:
            logits = model(inputs)

            sign = -1.0 if targeted else 1.0
            loss = sign * ep.crossentropy(logits, labels).sum()
            is_adv = criterion_(inputs, logits)

            return loss, is_adv

        grad_and_is_adversarial = ep.value_and_grad_fn(x, loss_fn, has_aux=True)

        delta = ep.zeros_like(x)

        epsilon = init_epsilon * ep.ones(x, len(x))
        worst_norm = flatten(ep.maximum(x, 1 - x)).square().sum(axis=-1).sqrt()

        best_l2 = worst_norm
        best_delta = delta
        adv_found = ep.zeros(x, len(x)).bool()

        for i in range(self.steps):
            x_adv = x + delta

            _, is_adversarial, gradients = grad_and_is_adversarial(x_adv, classes)
            gradients = normalize_l2_norms(gradients)

            l2 = ep.norms.l2(flatten(delta), axis=-1)
            is_smaller = l2 < best_l2

            is_both = ep.logical_and(is_adversarial, is_smaller)
            adv_found = ep.logical_or(adv_found, is_adversarial)
            best_l2 = ep.where(is_both, l2, best_l2)

            best_delta = ep.where(atleast_kd(is_both, x.ndim), delta, best_delta)

            # perform cosine annealing of LR starting from 1.0 to 0.01
            delta = delta + atleast_kd(stepsize, x.ndim) * gradients
            stepsize = (
                0.01 + (stepsize - 0.01) * (1 + math.cos(math.pi * i / self.steps)) / 2
            )

            epsilon = epsilon * ep.where(is_adversarial, 1 - self.gamma, 1 + self.gamma)
            epsilon = ep.minimum(epsilon, worst_norm)

            # do step
            delta = delta + atleast_kd(stepsize, x.ndim) * gradients

            # clip to valid bounds
            delta = (
                delta
                * atleast_kd(epsilon, x.ndim)
                / delta.square().sum(axis=(1, 2, 3), keepdims=True).sqrt()
            )
            delta = ep.clip(x + delta, *model.bounds) - x

        x_adv = x + delta

        return restore_type(x_adv)
