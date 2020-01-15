import numpy as np
import eagerpy as ep
from functools import partial

from ..utils import flatten
from ..utils import atleast_kd


class EADAttack:
    "EAD Attack with EN Decision Rule"

    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        inputs,
        labels,
        *,
        target_classes=None,
        binary_search_steps=9,
        max_iterations=10000,
        confidence=0,
        initial_learning_rate=1e-2,
        regularization=1e-2,
        initial_const=1e-3,
        abort_early=True,
        decision_rule="EN",
    ):
        x_0 = ep.astensor(inputs)
        N = len(x_0)

        assert decision_rule in ("EN", "L1")

        targeted = target_classes is not None
        if targeted:
            labels = None
            target_classes = ep.astensor(target_classes)
            assert target_classes.shape == (N,)
            is_adv = partial(
                targeted_is_adv, target_classes=target_classes, confidence=confidence
            )
        else:
            labels = ep.astensor(labels)
            assert labels.shape == (N,)
            is_adv = partial(untargeted_is_adv, labels=labels, confidence=confidence)

        min_, max_ = self.model.bounds()

        rows = np.arange(N)

        def loss_fun(y_k: ep.Tensor, consts: ep.Tensor) -> ep.Tensor:
            assert y_k.shape == x_0.shape
            assert consts.shape == (N,)

            logits = ep.astensor(self.model.forward(y_k.tensor))

            if targeted:
                c_minimize = best_other_classes(logits, target_classes)
                c_maximize = target_classes
            else:
                c_minimize = labels
                c_maximize = best_other_classes(logits, labels)

            is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)
            is_adv_loss = is_adv_loss + confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts

            squared_norms = flatten(y_k - x_0).square().sum(axis=-1)
            loss = is_adv_loss.sum() + squared_norms.sum()
            return loss, (y_k, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x_0, loss_fun, has_aux=True)

        consts = initial_const * np.ones((N,))
        lower_bounds = np.zeros((N,))
        upper_bounds = np.inf * np.ones((N,))

        best_advs = ep.zeros_like(x_0)
        best_advs_norms = ep.ones(x_0, (N,)) * np.inf

        # the binary search searches for the smallest consts that produce adversarials
        for binary_search_step in range(binary_search_steps):
            if (
                binary_search_step == binary_search_steps - 1
                and binary_search_steps >= 10
            ):
                # in the last iteration, repeat the search once
                consts = np.minimum(upper_bounds, 1e10)

            # create a new optimizer find the delta that minimizes the loss
            # TODO: rewrite this once eagerpy supports .copy()
            x_k = x_0  # ep.zeros_like(x_0) + x_0
            y_k = x_0  # ep.zeros_like(x_0) + x_0

            found_advs = np.full(
                (N,), fill_value=False
            )  # found adv with the current consts
            loss_at_previous_check = np.inf

            consts_ = ep.from_numpy(x_0, consts.astype(np.float32))

            for iteration in range(max_iterations):
                # square-root learning rate decay
                learning_rate = (
                    initial_learning_rate * (1.0 - iteration / max_iterations) ** 0.5
                )

                loss, (x, logits), gradient = loss_aux_and_grad(x_k, consts_)

                x_k_old = x_k
                x_k = project_shrinkage_thresholding(
                    y_k - learning_rate * gradient, x_0, regularization, min_, max_
                )
                y_k = x_k + iteration / (iteration + 3) - (x_k - x_k_old)

                if abort_early and iteration % (np.ceil(max_iterations / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if not (loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has been no progress
                    loss_at_previous_check = loss

                found_advs_iter = is_adv(logits)

                best_advs, best_advs_norms = apply_decision_rule(
                    decision_rule,
                    regularization,
                    best_advs,
                    best_advs_norms,
                    x_k,
                    x_0,
                    found_advs_iter,
                )

                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())

            upper_bounds = np.where(found_advs, consts, upper_bounds)
            lower_bounds = np.where(found_advs, lower_bounds, consts)

            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = np.where(
                np.isinf(upper_bounds), consts_exponential_search, consts_binary_search
            )

        return best_advs.tensor


def untargeted_is_adv(logits: ep.Tensor, labels: ep.Tensor, confidence) -> ep.Tensor:
    logits = logits + ep.onehot_like(logits, labels, value=confidence)
    classes = logits.argmax(axis=-1)
    return classes != labels


def targeted_is_adv(
    logits: ep.Tensor, target_classes: ep.Tensor, confidence
) -> ep.Tensor:
    logits = logits - ep.onehot_like(logits, target_classes, value=confidence)
    classes = logits.argmax(axis=-1)
    return classes == target_classes


def best_other_classes(logits, exclude):
    other_logits = logits - ep.onehot_like(logits, exclude, value=np.inf)
    return other_logits.argmax(axis=-1)


def apply_decision_rule(
    decision_rule: str,
    beta: float,
    best_advs: ep.Tensor,
    best_advs_norms: ep.Tensor,
    x_k: ep.Tensor,
    x_0: ep.Tensor,
    found_advs: ep.Tensor,
):
    if decision_rule == "EN":
        norms = beta * flatten(x_k - x_0).abs().sum(axis=-1) + flatten(
            x_k - x_0
        ).square().sum(axis=-1)
    elif decision_rule == "L1":
        norms = flatten(x_k - x_0).abs().sum(axis=-1)
    else:
        raise ValueError("invalid decision rule")

    new_best = (norms < best_advs_norms).float32() * found_advs.float32()
    new_best = atleast_kd(new_best, best_advs.ndim)
    best_advs = new_best * x_k + (1 - new_best) * best_advs
    best_advs_norms = ep.minimum(norms, best_advs_norms)

    return best_advs, best_advs_norms


def project_shrinkage_thresholding(
    z: ep.Tensor, x0: ep.Tensor, regularization: float, min_: float, max_: float
) -> ep.Tensor:
    """Performs the element-wise projected shrinkage-thresholding
    operation"""

    upper_mask = (z - x0 > regularization).float32()
    lower_mask = (z - x0 < -regularization).float32()

    projection = (
        (1.0 - upper_mask - lower_mask) * x0
        + upper_mask * ep.minimum(z - regularization, max_)
        + lower_mask * ep.maximum(z + regularization, min_)
    )

    return projection
