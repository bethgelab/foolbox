import logging

from .base import Attack
from .base import call_decorator

import numpy as np

class NewtonFoolAttack(Attack):

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True):
        """

        Implementaton according to "Objective Metrics and Gradient Descent Algorithms for
        Adversarial Examples in Machine Learning" from Uyeong Jang et al.
        Paper: https://andrewxiwu.github.io/public/papers/2017/JWJ17-objective-metrics-and-gradient-descent-based-algorithms-for-adversarial-examples-in-machine-learning.pdf

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
        """
        print(np.argmax(input_or_adv))
        pass


