from typing import Union, Tuple, Any, Optional
import numpy as np
import eagerpy as ep
import torch

from ..devutils import flatten
from ..devutils import atleast_kd


from ..models import Model

from ..distances import l2

from ..criteria import Misclassification
from ..criteria import TargetedMisclassification

from .base import MinimizationAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs
from .base import verify_input_bounds


class iHL_RFAttack(MinimizationAttack):
    """Implementation of the improved Hasofer-Lind, Rackwitz-Fiessler Attack.

    Args:
        steps : Number of optimization steps.
        confidence : Confidence required for an example to be marked as adversarial (just considered in the loss to go behind the frontier).
            Controls the gap between example and decision boundary.
        tau : multiplicative decrease of linear stepsize in Armijo's rule.
        smooth : growing factor in penalization term.
        omega : Armijo's factor, factor by which we want to assure we reduce enough.
        abort_early : Stop inner search as soons as an adversarial example has been found.
        min_steps : if abort early dont stop before a minimum step.


    """

    distance = l2

    def __init__(
        self,
        steps: int = 50,
        confidence: float = 0.1,
        tau: float = 0.1,
        smooth: float = 1.2,
        omega: float = 10e-4,
        abort_early: bool = True,
        min_steps: int = 25,
    ):
        self.steps = steps
        self.confidence = confidence
        self.tau = tau
        self.omega = omega
        self.smooth = smooth
        self.abort_early = abort_early
        self.min_steps = min_steps

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
        inputs_, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs
        verify_input_bounds(inputs_, model)

        N = len(inputs_)
        original_shape = inputs_.shape
        im_shape = (len(inputs_[0]), len(inputs_[0][0]), len(inputs_[0][0][0]))
        inputs_ = flatten(inputs_)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels

        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes

        else:
            raise ValueError("unsupported criterion")

        def is_adversarial(perturbed: ep.Tensor, classes: ep.Tensor) -> ep.Tensor:
            if not (targeted):
                return (
                    model(ep.reshape(perturbed, original_shape)).argmax(axis=-1)
                    != classes
                )
            else:
                return (
                    model(ep.reshape(perturbed, original_shape)).argmax(axis=-1)
                    == classes
                )

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        def loss_fun(
            delta: ep.Tensor, classes: ep.Tensor = classes
        ) -> Tuple[ep.Tensor, ep.Tensor]:

            """
            The loss used to study adverariality. If <0, then the image is adversarial
            """

            row = len(classes)
            rows = range(row)
            x = ep.reshape(delta, (len(classes), im_shape[0], im_shape[1], im_shape[2]))
            mod = model(x)
            logits = ep.softmax(mod)

            if targeted:
                c_minimize = best_other_classes(logits, classes)
                c_maximize = classes  # target_classes
            else:
                c_minimize = classes  # labels
                c_maximize = best_other_classes(logits, classes)

            loss = (
                logits[rows, c_minimize] - logits[rows, c_maximize]
            ) + self.confidence

            return loss.sum(), loss

        loss_aux_and_grad = ep.value_and_grad_fn(inputs_, loss_fun, has_aux=True)
        best_advs = ep.zeros_like(inputs_)
        best_advs_norms = ep.full(inputs_, (N,), ep.inf)

        # find the best stepsize thanks to the armijo's rule
        def pred_stepsize(
            self,
            x: ep.Tensor,
            desc_dir: ep.Tensor,
            loss_x: ep.Tensor,
            grad_x: ep.Tensor,
        ) -> ep.Tensor:
            step_armijo = ep.full_like(loss_x, 0)
            norm_x = x.norms.lp(p=2, axis=-1)
            grad_x_norm = grad_x.norms.lp(p=2, axis=-1)
            norm_grad = ep.where(
                grad_x_norm == 0, 1, grad_x_norm
            )  # assure that we dont divide by 0
            sigm = ep.maximum((self.smooth * norm_x) / norm_grad, (self.sigma + 1))
            res = []
            for i in range(N):
                res.append(ep.full_like(grad_x[0], sigm[i].item()))
            sigma_imsize = ep.stack(res)
            self.sigma = sigm
            mer = merit(sigm, norm_x, loss_x)
            grad_mer = grad_merit(sigma_imsize, x, loss_x, grad_x)
            cond = (desc_dir * grad_mer).sum(axis=-1) * self.omega
            found_armijo = ep.zeros_like(loss_x)
            k = 0
            while found_armijo.sum() < (N) and ((self.tau**k) > 10e-4):
                x_step = x + desc_dir * self.tau**k
                _, loss_step = loss_fun(x_step + inputs_, classes)
                norm_step = x_step.norms.lp(p=2, axis=-1)
                respect_armijo = ep.where(
                    merit(sigm, norm_step, loss_step) - mer > cond * self.tau**k, 0, 1
                )
                alr_armijo = ep.where(respect_armijo == 0, 0, self.tau**k)
                step_armijo = ep.where(
                    step_armijo == 0, step_armijo + alr_armijo, step_armijo
                )
                found_armijo = ep.where(step_armijo == 0, 0, 1)
                k += 1
            step_armijo = ep.where(step_armijo == 0, self.tau ** (k - 1), step_armijo)
            return step_armijo

        # gamma = -(Loss - <Grad(Loss), u>) / ||Grad(Loss)||^2
        # Compute the current direction Dir = gamma Grad(G) - u
        def descent_dir(x: ep.Tensor, grad: ep.Tensor, loss_x: ep.Tensor) -> ep.Tensor:
            norm_grad = grad.norms.lp(p=2, axis=-1)
            norm_grad = ep.where(
                norm_grad == ep.zeros_like(norm_grad),
                ep.ones_like(norm_grad),
                norm_grad,
            )
            gamma = ((grad * x).sum(axis=-1) - loss_x) / (norm_grad**2)
            res = []
            for i in range(len(gamma)):
                res.append(gamma[i] * grad[i] - x[i])

            return ep.stack(res)

        delta = ep.zeros_like(inputs_)
        found_advs = ep.full_like(classes, fill_value=False)
        found_advs = found_advs != 0
        self.sigma = ep.full(inputs_, (N,), 1)
        for step in range(self.steps):
            _, loss, gradient = loss_aux_and_grad(delta + inputs_)
            desc_dir = descent_dir(delta, gradient, loss)
            stepsize = pred_stepsize(
                self=self, x=delta, desc_dir=desc_dir, loss_x=loss, grad_x=gradient
            )
            next_step = []
            for i in range(N):
                next_step.append(desc_dir[i] * stepsize[i])
            next_step = ep.stack(next_step)
            delta += next_step
            found_advs_iter = is_adversarial(delta + inputs_, classes)
            found_advs = ep.logical_or(found_advs, found_advs_iter)

            norms = delta.norms.l2(axis=-1)
            closer = norms < best_advs_norms
            new_best = ep.logical_and(closer, found_advs_iter)

            new_best_ = atleast_kd(
                new_best, best_advs.ndim
            )  # reshape new_best dim + best_advs dim - new_best dim
            best_advs = ep.where(new_best_, delta + inputs_, best_advs)
            found_advs_ = atleast_kd(found_advs, best_advs.ndim)
            best_advs = ep.where(found_advs_, best_advs, delta + inputs_)
            # best_advs = ep.where(ep.astensor(found_advs),best_advs,delta+inputs_)
            best_advs_norms = ep.where(new_best, norms, best_advs_norms)

            if self.abort_early and ep.all(found_advs) and step > self.min_steps:
                break

        return restore_type(ep.reshape(best_advs, original_shape))


def best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    other_logits = logits - ep.onehot_like(logits, exclude, value=np.inf)
    res = other_logits.argmax(axis=-1)

    return res


# define the merit function and her gradient for the armijo's rule
def merit(sigma: ep.Tensor, norm_x: ep.Tensor, loss_x: ep.Tensor) -> ep.Tensor:
    return 0.5 * (norm_x) ** 2 + sigma * ep.abs(loss_x)


def grad_merit(
    sigma: ep.Tensor, x: ep.Tensor, loss_x: ep.Tensor, grad_x: ep.Tensor
) -> ep.Tensor:
    sign = ep.sign(loss_x)
    sign = atleast_kd(sign, grad_x.ndim)
    sign = ep.sign(loss_x)
    res = []
    for i in range(len(sign)):
        res.append(ep.full_like(grad_x[0], sign[i].item()))
    sign = ep.stack(res)
    return x + sigma * grad_x * sign
