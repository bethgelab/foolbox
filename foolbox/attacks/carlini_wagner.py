# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import logging

from .base import Attack
from .base import call_decorator
from ..utils import onehot_like


class CarliniWagnerL2Attack(Attack):
    """The L2 version of the Carlini & Wagner attack.

    This attack is described in [1]_. This implementation
    is based on the reference implementation by Carlini [2]_.
    For bounds ≠ (0, 1), it differs from [2]_ because we
    normalize the squared L2 loss with the bounds.

    References
    ----------
    .. [1] Nicholas Carlini, David Wagner: "Towards Evaluating the
           Robustness of Neural Networks", https://arxiv.org/abs/1608.04644
    .. [2] https://github.com/carlini/nn_robust_attacks

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search_steps=5, max_iterations=1000,
                 confidence=0, learning_rate=5e-3,
                 initial_const=1e-2, abort_early=True):

        """The L2 version of the Carlini & Wagner attack.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search_steps : int
            The number of steps for the binary search used to
            find the optimal tradeoff-constant between distance and confidence.
        max_iterations : int
            The maximum number of iterations. Larger values are more
            accurate; setting it too small will require a large learning rate
            and will produce poor results.
        confidence : int or float
            Confidence of adversarial examples: a higher value produces
            adversarials that are further away, but more strongly classified
            as adversarial.
        learning_rate : float
            The learning rate for the attack algorithm. Smaller values
            produce better results but take longer to converge.
        initial_const : float
            The initial tradeoff-constant to use to tune the relative
            importance of distance and confidence. If `binary_search_steps`
            is large, the initial constant is not important.
        abort_early : bool
            If True, Adam will be aborted if the loss hasn't decreased
            for some time (a tenth of max_iterations).

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            logging.fatal('Applied gradient-based attack to model that '
                          'does not provide gradients.')
            return

        min_, max_ = a.bounds()

        def to_attack_space(x):
            # map from [min_, max_] to [-1, +1]
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = (x - a) / b

            # from [-1, +1] to approx. (-1, +1)
            x = x * 0.999999

            # from (-1, +1) to (-inf, +inf)
            return np.arctanh(x)

        def to_model_space(x):
            """Transforms an input from the attack space
            to the model space. This transformation and
            the returned gradient are elementwise."""

            # from (-inf, +inf) to (-1, +1)
            x = np.tanh(x)

            grad = 1 - np.square(x)

            # map from (-1, +1) to (min_, max_)
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = x * b + a

            grad = grad * b
            return x, grad

        # variables representing inputs in attack space will be
        # prefixed with att_
        att_original = to_attack_space(a.unperturbed)

        # will be close but not identical to a.unperturbed
        reconstructed_original, _ = to_model_space(att_original)

        # the binary search finds the smallest const for which we
        # find an adversarial
        const = initial_const
        lower_bound = 0
        upper_bound = np.inf

        for binary_search_step in range(binary_search_steps):
            if binary_search_step == binary_search_steps - 1 and \
                    binary_search_steps >= 10:
                # in the last binary search step, use the upper_bound instead
                # TODO: find out why... it's not obvious why this is useful
                const = min(1e10, upper_bound)

            logging.info('starting optimization with const = {}'.format(const))

            att_perturbation = np.zeros_like(att_original)

            # create a new optimizer to minimize the perturbation
            optimizer = AdamOptimizer(att_perturbation.shape)

            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf

            for iteration in range(max_iterations):
                x, dxdp = to_model_space(att_original + att_perturbation)
                logits, is_adv = a.forward_one(x)
                loss, dldx = self.loss_function(
                    const, a, x, logits, reconstructed_original,
                    confidence, min_, max_)

                logging.info('loss: {}; best overall distance: {}'.format(loss, a.distance))

                # backprop the gradient of the loss w.r.t. x further
                # to get the gradient of the loss w.r.t. att_perturbation
                assert dldx.shape == x.shape
                assert dxdp.shape == x.shape
                # we can do a simple elementwise multiplication, because
                # grad_x_wrt_p is a matrix of elementwise derivatives
                # (i.e. each x[i] w.r.t. p[i] only, for all i) and
                # grad_loss_wrt_x is a real gradient reshaped as a matrix
                gradient = dldx * dxdp

                att_perturbation += optimizer(gradient, learning_rate)

                if is_adv:
                    # this binary search step can be considered a success
                    # but optimization continues to minimize perturbation size
                    found_adv = True

                if abort_early and \
                        iteration % (np.ceil(max_iterations / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if not (loss <= .9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = loss

            if found_adv:
                logging.info('found adversarial with const = {}'.format(const))
                upper_bound = const
            else:
                logging.info('failed to find adversarial '
                             'with const = {}'.format(const))
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2

    @classmethod
    def loss_function(cls, const, a, x, logits, reconstructed_original,
                      confidence, min_, max_):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        targeted = a.target_class() is not None
        if targeted:
            c_minimize = cls.best_other_class(logits, a.target_class())
            c_maximize = a.target_class()
        else:
            c_minimize = a.original_class
            c_maximize = cls.best_other_class(logits, a.original_class)

        is_adv_loss = logits[c_minimize] - logits[c_maximize]

        # is_adv is True as soon as the is_adv_loss goes below 0
        # but sometimes we want additional confidence
        is_adv_loss += confidence
        is_adv_loss = max(0, is_adv_loss)

        s = max_ - min_
        squared_l2_distance = np.sum((x - reconstructed_original)**2) / s**2
        total_loss = squared_l2_distance + const * is_adv_loss

        # calculate the gradient of total_loss w.r.t. x
        logits_diff_grad = np.zeros_like(logits)
        logits_diff_grad[c_minimize] = 1
        logits_diff_grad[c_maximize] = -1
        is_adv_loss_grad = a.backward_one(logits_diff_grad, x)
        assert is_adv_loss >= 0
        if is_adv_loss == 0:
            is_adv_loss_grad = 0

        squared_l2_distance_grad = (2 / s**2) * (x - reconstructed_original)

        total_loss_grad = squared_l2_distance_grad + const * is_adv_loss_grad
        return total_loss, total_loss_grad

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`."""
        other_logits = logits - onehot_like(logits, exclude, value=np.inf)
        return np.argmax(other_logits)


class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable.

    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized

    """

    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def __call__(self, gradient, learning_rate,
                 beta1=0.9, beta2=0.999, epsilon=10e-8):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.

        Parameters
        ----------
        gradient : `np.ndarray`
            the gradient of the loss w.r.t. to the variable
        learning_rate: float
            the learning rate in the current iteration
        beta1: float
            decay rate for calculating the exponentially
            decaying average of past gradients
        beta2: float
            decay rate for calculating the exponentially
            decaying average of past squared gradients
        epsilon: float
            small value to avoid division by zero

        """

        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient**2

        bias_correction_1 = 1 - beta1**self.t
        bias_correction_2 = 1 - beta2**self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
