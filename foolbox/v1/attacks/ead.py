import numpy as np
import logging

from .base import Attack
from .base import call_decorator
from ...utils import onehot_like


class EADAttack(Attack):
    """Gradient based attack which uses an elastic-net regularization [1].
    This implementation is based on the attacks description [1] and its
    reference implementation [2].

    References
    ----------
    .. [1] Pin-Yu Chen (*), Yash Sharma (*), Huan Zhang, Jinfeng Yi,
           Cho-Jui Hsieh, "EAD: Elastic-Net Attacks to Deep Neural
           Networks via Adversarial Examples",
           https://arxiv.org/abs/1709.04114

    .. [2] Pin-Yu Chen (*), Yash Sharma (*), Huan Zhang, Jinfeng Yi,
           Cho-Jui Hsieh, "Reference Implementation of 'EAD: Elastic-Net
           Attacks to Deep Neural Networks via Adversarial Examples'",
           https://github.com/ysharma1126/EAD_Attack/blob/master/en_attack.py
    """

    @call_decorator
    def __call__(
        self,
        input_or_adv,
        label=None,
        unpack=True,
        binary_search_steps=5,
        max_iterations=1000,
        confidence=0,
        initial_learning_rate=1e-2,
        regularization=1e-2,
        initial_const=1e-2,
        abort_early=True,
    ):

        """Gradient based attack which sues an elastic-net regularization.

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
        initial_learning_rate : float
            The initial learning rate for the attack algorithm. Smaller values
            produce better results but take longer to converge. During the
            attack a square-root decay in the learning rate is performed.
        initial_const : float
            The initial tradeoff-constant to use to tune the relative
            importance of distance and confidence. If `binary_search_steps`
            is large, the initial constant is not important.
        regularization : float
            The L1 regularization parameter (also called beta). A value of `0`
            corresponds to the :class:`attacks.CarliniWagnerL2Attack` attack.
        abort_early : bool
            If True, Adam will be aborted if the loss hasn't decreased
            for some time (a tenth of max_iterations).

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            logging.fatal(
                "Applied gradient-based attack to model that "
                "does not provide gradients."
            )
            return

        min_, max_ = a.bounds()

        # variables representing inputs in attack space will be
        # prefixed with att_
        att_original = a.unperturbed

        # the binary search finds the smallest const for which we
        # find an adversarial
        const = initial_const
        lower_bound = 0
        upper_bound = np.inf

        for binary_search_step in range(binary_search_steps):
            if (
                binary_search_step == binary_search_steps - 1
                and binary_search_steps >= 10
            ):
                # in the last binary search step, use the upper_bound instead
                # TODO: find out why... it's not obvious why this is useful
                const = min(1e10, upper_bound)

            logging.info("starting optimization with const = {}".format(const))

            x = att_original.copy()
            # corresponds to x^(k-1)
            x_prev = None
            y = att_original.copy()

            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf

            for iteration in range(max_iterations):
                # square-root learning rate decay
                learning_rate = (
                    initial_learning_rate * (1.0 - iteration / max_iterations) ** 0.5
                )

                # store x from previous iteration (k-1) as x^(k-1)
                x_prev = x.copy()

                logits, is_adv = a.forward_one(y)
                loss, gradient = self.loss_function(
                    const, a, y, logits, att_original, confidence, min_, max_
                )

                logging.info(
                    "loss: {}; best overall distance: {}".format(loss, a.distance)
                )

                # backprop the gradient of the loss w.r.t. x further
                # to get the gradient of the loss w.r.t. att_perturbation
                assert gradient.shape == x.shape

                x = self.project_shrinkage_thresholding(
                    y - learning_rate * gradient,
                    att_original,
                    regularization,
                    min_,
                    max_,
                )
                y = x + iteration / (iteration + 3.0) * (x - x_prev)

                # clip the slack variable to make sure that it is still
                # in valid bounds for the model
                y = np.clip(y, min_, max_)

                if is_adv:
                    # this binary search step can be considered a success
                    # but optimization continues to minimize perturbation size
                    found_adv = True

                if abort_early and iteration % (np.ceil(max_iterations / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if not (loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = loss

            if found_adv:
                logging.info("found adversarial with const = {}".format(const))
                upper_bound = const
            else:
                logging.info(
                    "failed to find adversarial " "with const = {}".format(const)
                )
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2

    @classmethod
    def loss_function(cls, const, a, x, logits, original, confidence, min_, max_):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        targeted = a.target_class is not None
        if targeted:
            c_minimize = cls.best_other_class(logits, a.target_class)
            c_maximize = a.target_class
        else:
            c_minimize = a.original_class
            c_maximize = cls.best_other_class(logits, a.original_class)

        is_adv_loss = logits[c_minimize] - logits[c_maximize]

        # is_adv is True as soon as the is_adv_loss goes below 0
        # but sometimes we want additional confidence
        is_adv_loss += confidence
        is_adv_loss = max(0, is_adv_loss)

        s = max_ - min_
        squared_l2_distance = np.sum((x - original) ** 2) / s ** 2

        total_loss = squared_l2_distance + const * is_adv_loss

        # calculate the gradient of total_loss w.r.t. x
        logits_diff_grad = np.zeros_like(logits)
        logits_diff_grad[c_minimize] = 1
        logits_diff_grad[c_maximize] = -1
        is_adv_loss_grad = a.backward_one(logits_diff_grad, x)
        assert is_adv_loss >= 0
        if is_adv_loss == 0:
            is_adv_loss_grad = 0

        squared_l2_distance_grad = (2 / s ** 2) * (x - original)

        total_loss_grad = squared_l2_distance_grad + const * is_adv_loss_grad
        return total_loss, total_loss_grad

    @classmethod
    def project_shrinkage_thresholding(cls, z, x0, regularization, min_, max_):
        """Performs the element-wise projected shrinkage-thresholding
        operation"""

        projection = x0.copy()

        upper_mask = z - x0 > regularization
        lower_mask = z - x0 < -regularization

        projection[upper_mask] = np.minimum(z - regularization, max_)[upper_mask]
        projection[lower_mask] = np.maximum(z + regularization, min_)[lower_mask]

        return projection

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`."""
        other_logits = logits - onehot_like(logits, exclude, value=np.inf)
        return np.argmax(other_logits)
