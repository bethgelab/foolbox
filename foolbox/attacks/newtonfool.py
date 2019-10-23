import logging

from .base import Attack
from .base import generator_decorator
from ..utils import softmax

import numpy as np


class NewtonFoolAttack(Attack):
    """Implements the NewtonFool Attack.

    The attack was introduced in [1]_.

    References
    ----------
    .. [1] Uyeong Jang et al., "Objective Metrics and Gradient Descent
           Algorithms for Adversarial Examples in Machine Learning",
           https://dl.acm.org/citation.cfm?id=3134635
   """

    @generator_decorator
    def as_generator(self, a, max_iter=100, eta=0.01):
        """
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
        max_iter : int
            The maximum number of iterations.
        eta : float
            the eta coefficient
        """

        if not a.has_gradient():
            logging.fatal(
                "Applied gradient-based attack to model that "
                "does not provide gradients."
            )

            return

        if a.target_class is not None:
            logging.fatal("NewtonFool is an untargeted adversarial attack.")
            return

        l2_norm = np.linalg.norm(a.unperturbed)
        min_, max_ = a.bounds()
        perturbed = a.unperturbed.copy()

        for i in range(max_iter):

            # (1) get the score and gradients
            logits, gradients, is_adversarial = yield from a.forward_and_gradient_one(
                perturbed
            )

            if is_adversarial:
                return

            score = np.max(softmax(logits))
            # instead of using the logits and the gradient of the logits,
            # we use a numerically stable implementation of the cross-entropy
            # and expect that the deep learning frameworks also use such a
            # stable implemenation to calculate the gradient
            # grad is calculated from CE but we want softmax
            # -> revert chain rule
            gradients = -gradients / score

            # (2) calculate gradient norm
            gradient_l2_norm = np.linalg.norm(gradients)

            # (3) calculate delta
            delta = self._delta(eta, l2_norm, score, gradient_l2_norm, a.num_classes())

            # delta = 0.01

            # (4) calculate & apply current pertubation
            current_pertubation = self._perturbation(delta, gradients, gradient_l2_norm)

            perturbed += current_pertubation
            perturbed = np.clip(perturbed, min_, max_)

    @staticmethod
    def _delta(eta, norm, score, gradient_norm, num_classes):
        a = eta * norm * gradient_norm
        b = score - 1.0 / num_classes
        return min(a, b)

    @staticmethod
    def _perturbation(delta, gradients, gradient_norm):
        direction = -((delta / (gradient_norm ** 2)) * gradients)
        return direction
