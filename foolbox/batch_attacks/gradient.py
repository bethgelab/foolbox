from __future__ import division
import numpy as np
from collections import Iterable
import logging
import abc

from .base import BatchAttack
from .base import generator_decorator


class SingleStepGradientBaseAttack(BatchAttack):
    """Common base class for single step gradient attacks."""

    @abc.abstractmethod
    def _gradient(self, a):
        raise NotImplementedError

    def _run(self, a, epsilons, max_epsilon):
        if not a.has_gradient():
            return

        x = a.unperturbed
        min_, max_ = a.bounds()

        gradient = yield from self._gradient(a)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, max_epsilon, num=epsilons + 1)[1:]
            decrease_if_first = True
        else:
            decrease_if_first = False

        for _ in range(2):  # to repeat with decreased epsilons if necessary
            for i, epsilon in enumerate(epsilons):
                perturbed = x + gradient * epsilon
                perturbed = np.clip(perturbed, min_, max_)

                _, is_adversarial = yield from a.forward_one(perturbed)
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

    @generator_decorator
    def as_generator(self, a, epsilons=1000, max_epsilon=1):
        """Perturbs the input with the gradient of the loss w.r.t. the input,
        gradually increasing the magnitude until the input is misclassified.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the gradient direction
            or number of step sizes between 0 and max_epsilon that should
            be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.

        """

        yield from self._run(a, epsilons=epsilons, max_epsilon=max_epsilon)

    def _gradient(self, a):
        min_, max_ = a.bounds()
        gradient = yield from a.gradient_one()
        gradient_norm = np.sqrt(np.mean(np.square(gradient)))
        gradient = gradient / (gradient_norm + 1e-8) * (max_ - min_)
        return gradient


GradientAttack.__call__.__doc__ = GradientAttack.as_generator.__doc__


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

    @generator_decorator
    def as_generator(self, a, epsilons=1000, max_epsilon=1):
        """Adds the sign of the gradient to the input, gradually increasing
        the magnitude until the input is misclassified.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the direction of the sign of
            the gradient or number of step sizes between 0 and max_epsilon
            that should be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.

        """

        yield from self._run(a, epsilons=epsilons, max_epsilon=max_epsilon)

    def _gradient(self, a):
        min_, max_ = a.bounds()
        gradient = yield from a.gradient_one()
        gradient = np.sign(gradient) * (max_ - min_)
        return gradient


GradientSignAttack.__call__.__doc__ = GradientSignAttack.as_generator.__doc__


FGSM = GradientSignAttack
