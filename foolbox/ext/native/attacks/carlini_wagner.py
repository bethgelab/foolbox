import numpy as np
import eagerpy as ep
from functools import partial

from ..utils import flatten
from ..utils import atleast_kd


class L2CarliniWagnerAttack:
    "Carlini Wagner L2 Attack"

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
        learning_rate=1e-2,
        initial_const=1e-3,
        abort_early=True,
    ):
        x = ep.astensor(inputs)
        N = len(x)

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

        bounds = self.model.bounds()
        to_attack_space = partial(_to_attack_space, bounds=bounds)
        to_model_space = partial(_to_model_space, bounds=bounds)

        x_attack = to_attack_space(x)
        reconstsructed_x = to_model_space(x_attack)

        rows = np.arange(N)

        def loss_fun(delta: ep.Tensor, consts: ep.Tensor) -> ep.Tensor:
            assert delta.shape == x_attack.shape
            assert consts.shape == (N,)

            x = to_model_space(x_attack + delta)
            logits = ep.astensor(self.model.forward(x.tensor))

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

            squared_norms = flatten(x - reconstsructed_x).square().sum(axis=-1)
            loss = is_adv_loss.sum() + squared_norms.sum()
            return loss, (x, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        consts = initial_const * np.ones((N,))
        lower_bounds = np.zeros((N,))
        upper_bounds = np.inf * np.ones((N,))

        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.ones(x, (N,)) * np.inf

        # the binary search searches for the smallest consts that produce adversarials
        for binary_search_step in range(binary_search_steps):
            if (
                binary_search_step == binary_search_steps - 1
                and binary_search_steps >= 10
            ):
                # in the last iteration, repeat the search once
                consts = np.minimum(upper_bounds, 1e10)

            # create a new optimizer find the delta that minimizes the loss
            delta = ep.zeros_like(x_attack)
            optimizer = AdamOptimizer(delta)

            found_advs = np.full(
                (N,), fill_value=False
            )  # found adv with the current consts
            loss_at_previous_check = np.inf

            consts_ = ep.from_numpy(x, consts.astype(np.float32))

            for iteration in range(max_iterations):
                loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_)
                delta += optimizer(gradient, learning_rate)

                if abort_early and iteration % (np.ceil(max_iterations / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if not (loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has been no progress
                    loss_at_previous_check = loss

                found_advs_iter = is_adv(logits)
                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())

                norms = flatten(perturbed - x).square().sum(axis=-1).sqrt()
                closer = norms < best_advs_norms
                new_best = closer.float32() * found_advs_iter.float32()

                best_advs = (
                    atleast_kd(new_best, best_advs.ndim) * perturbed
                    + (1 - atleast_kd(new_best, best_advs.ndim)) * best_advs
                )
                best_advs_norms = new_best * norms + (1 - new_best) * best_advs_norms

            upper_bounds = np.where(found_advs, consts, upper_bounds)
            lower_bounds = np.where(found_advs, lower_bounds, consts)

            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = np.where(
                np.isinf(upper_bounds), consts_exponential_search, consts_binary_search
            )

        return best_advs.tensor


class AdamOptimizer:
    def __init__(self, x):
        self.m = ep.zeros_like(x)
        self.v = ep.zeros_like(x)
        self.t = 0

    def __call__(self, gradient, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2

        bias_correction_1 = 1 - beta1 ** self.t
        bias_correction_2 = 1 - beta2 ** self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -learning_rate * m_hat / (ep.sqrt(v_hat) + epsilon)


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


def _to_attack_space(x: ep.Tensor, *, bounds: tuple) -> ep.Tensor:
    min_, max_ = bounds
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = (x - a) / b  # map from [min_, max_] to [-1, +1]
    x = x * 0.999999  # from [-1, +1] to approx. (-1, +1)
    x = x.arctanh()  # from (-1, +1) to (-inf, +inf)
    return x


def _to_model_space(x: ep.Tensor, *, bounds) -> ep.Tensor:
    min_, max_ = bounds
    x = x.tanh()  # from (-inf, +inf) to (-1, +1)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = x * b + a  # map from (-1, +1) to (min_, max_)
    return x
