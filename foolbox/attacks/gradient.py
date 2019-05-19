from __future__ import division
import numpy as np
from collections import Iterable
import logging
import abc

from .base import Attack
from .base import call_decorator


class SingleStepGradientBaseAttack(Attack):
    """Common base class for single step gradient attacks."""

    @abc.abstractmethod
    def _gradient(self, a):
        raise NotImplementedError

    def _run(self, a, epsilons, max_epsilon):
        if not a.has_gradient():
            return

        x = a.unperturbed
        min_, max_ = a.bounds()

        gradient = self._gradient(a)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, max_epsilon, num=epsilons + 1)[1:]
            decrease_if_first = True
        else:
            decrease_if_first = False

        for _ in range(2):  # to repeat with decreased epsilons if necessary
            for i, epsilon in enumerate(epsilons):
                perturbed = x + gradient * epsilon
                perturbed = np.clip(perturbed, min_, max_)

                _, is_adversarial = a.forward_one(perturbed)
                if is_adversarial:
                    if decrease_if_first and i < 20:
                        logging.info('repeating attack with smaller epsilons')
                        break
                    return

            max_epsilon = epsilons[i]
            epsilons = np.linspace(0, max_epsilon, num=20 + 1)[1:]


class GradientAttack(SingleStepGradientBaseAttack):
    """Perturbs the input with the gradient of the loss w.r.t. the input,
    gradually increasing the magnitude until the input is misclassified.

    Does not do anything if the model does not have a gradient.

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 epsilons=1000, max_epsilon=1):

        """Perturbs the input with the gradient of the loss w.r.t. the input,
        gradually increasing the magnitude until the input is misclassified.

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
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the gradient direction
            or number of step sizes between 0 and max_epsilon that should
            be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        return self._run(a, epsilons=epsilons, max_epsilon=max_epsilon)

    def _gradient(self, a):
        min_, max_ = a.bounds()
        gradient = a.gradient_one()
        gradient_norm = np.sqrt(np.mean(np.square(gradient)))
        gradient = gradient / (gradient_norm + 1e-8) * (max_ - min_)
        return gradient


class GradientSignAttack(SingleStepGradientBaseAttack):
    """Adds the sign of the gradient to the input, gradually increasing
    the magnitude until the input is misclassified. This attack is
    often referred to as Fast Gradient Sign Method and was introduced
    in [1]_.

    Does not do anything if the model does not have a gradient.

    References
    ----------
    .. [1] Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy,
           "Explaining and Harnessing Adversarial Examples",
           https://arxiv.org/abs/1412.6572
    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 epsilons=1000, max_epsilon=1):

        """Adds the sign of the gradient to the input, gradually increasing
        the magnitude until the input is misclassified.

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
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the direction of the sign of
            the gradient or number of step sizes between 0 and max_epsilon
            that should be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        return self._run(a, epsilons=epsilons, max_epsilon=max_epsilon)

    def _gradient(self, a):
        min_, max_ = a.bounds()
        gradient = a.gradient_one()
        gradient = np.sign(gradient) * (max_ - min_)
        return gradient


FGSM = GradientSignAttack
