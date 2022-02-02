from typing import Union, Tuple, Optional, Any
import math
import eagerpy as ep

from ..models import Model

from ..criteria import Misclassification, TargetedMisclassification

from ..distances import l2

from ..devutils import atleast_kd, flatten

from .base import MinimizationAttack
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs


def normalize_gradient_l2_norms(grad: ep.Tensor) -> ep.Tensor:
    norms = ep.norms.l2(flatten(grad), -1)

    # remove zero gradients
    grad = ep.where(
        atleast_kd(norms == 0, grad.ndim), ep.normal(grad, shape=grad.shape), grad
    )
    # calculate norms again for previously vanishing elements
    norms = ep.norms.l2(flatten(grad), -1)

    norms = ep.maximum(norms, 1e-12)  # avoid division by zero
    factor = 1 / norms
    factor = atleast_kd(factor, grad.ndim)
    return grad * factor


class DDNAttack(MinimizationAttack):
    """The Decoupled Direction and Norm L2 adversarial attack. [#Rony18]_

    Args:
        init_epsilon : Initial value for the norm/epsilon ball.
        steps : Number of steps for the optimization.
        gamma : Factor by which the norm will be modified: new_norm = norm * (1 + or - gamma).

    References:
        .. [#Rony18] Jérôme Rony, Luiz G. Hafemann, Luiz S. Oliveira, Ismail Ben Ayed,
            Robert Sabourin, Eric Granger, "Decoupling Direction and Norm for
            Efficient Gradient-Based L2 Adversarial Attacks and Defenses",
            https://arxiv.org/abs/1811.09600
    """

    distance = l2

    def __init__(
        self, *, init_epsilon: float = 1.0, steps: int = 100, gamma: float = 0.05,
    ):
        self.init_epsilon = init_epsilon
        self.steps = steps
        self.gamma = gamma

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

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

        max_stepsize = 1.0
        min_, max_ = model.bounds

        def loss_fn(
            inputs: ep.Tensor, labels: ep.Tensor
        ) -> Tuple[ep.Tensor, ep.Tensor]:
            logits = model(inputs)

            sign = -1.0 if targeted else 1.0
            loss = sign * ep.crossentropy(logits, labels).sum()

            return loss, logits

        grad_and_logits = ep.value_and_grad_fn(x, loss_fn, has_aux=True)

        delta = ep.zeros_like(x)

        epsilon = self.init_epsilon * ep.ones(x, len(x))
        worst_norm = ep.norms.l2(flatten(ep.maximum(x - min_, max_ - x)), -1)

        best_l2 = worst_norm
        best_delta = delta
        adv_found = ep.zeros(x, len(x)).bool()

        for i in range(self.steps):
            # perform cosine annealing of LR starting from 1.0 to 0.01
            stepsize = (
                0.01
                + (max_stepsize - 0.01) * (1 + math.cos(math.pi * i / self.steps)) / 2
            )

            x_adv = x + delta

            _, logits, gradients = grad_and_logits(x_adv, classes)
            gradients = normalize_gradient_l2_norms(gradients)
            is_adversarial = criterion_(x_adv, logits)

            l2 = ep.norms.l2(flatten(delta), axis=-1)
            is_smaller = l2 <= best_l2

            is_both = ep.logical_and(is_adversarial, is_smaller)
            adv_found = ep.logical_or(adv_found, is_adversarial)
            best_l2 = ep.where(is_both, l2, best_l2)

            best_delta = ep.where(atleast_kd(is_both, x.ndim), delta, best_delta)

            # do step
            delta = delta + stepsize * gradients

            epsilon = epsilon * ep.where(
                is_adversarial, 1.0 - self.gamma, 1.0 + self.gamma
            )
            epsilon = ep.minimum(epsilon, worst_norm)

            # project to epsilon ball
            delta *= atleast_kd(epsilon / ep.norms.l2(flatten(delta), -1), x.ndim)

            # clip to valid bounds
            delta = ep.clip(x + delta, *model.bounds) - x

        x_adv = x + best_delta

        return restore_type(x_adv)
