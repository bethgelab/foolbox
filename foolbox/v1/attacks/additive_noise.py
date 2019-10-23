from abc import abstractmethod
from collections import Iterable

import numpy as np

from .base import Attack
from .base import call_decorator
from .. import nprng


class AdditiveNoiseAttack(Attack):
    """Base class for attacks that add random noise to an input.

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True, epsilons=1000):
        """Adds uniform or Gaussian noise to the input, gradually increasing
        the standard deviation until the input is misclassified.

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
            Either Iterable of noise levels or number of noise levels
            between 0 and 1 that should be tried.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        x = a.unperturbed
        bounds = a.bounds()
        min_, max_ = bounds

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            noise = self._sample_noise(epsilon, x, bounds)
            perturbed = x + epsilon * noise
            perturbed = np.clip(perturbed, min_, max_)

            _, is_adversarial = a.forward_one(perturbed)
            if is_adversarial:
                return

    @abstractmethod
    def _sample_noise(self):
        raise NotImplementedError


class AdditiveUniformNoiseAttack(AdditiveNoiseAttack):
    """Adds uniform noise to the input, gradually increasing
    the standard deviation until the input is misclassified.

    """

    def _sample_noise(self, epsilon, x, bounds):
        min_, max_ = bounds
        w = epsilon * (max_ - min_)
        noise = nprng.uniform(-w, w, size=x.shape)
        noise = noise.astype(x.dtype)
        return noise


class AdditiveGaussianNoiseAttack(AdditiveNoiseAttack):
    """Adds Gaussian noise to the input, gradually increasing
    the standard deviation until the input is misclassified.

    """

    def _sample_noise(self, epsilon, x, bounds):
        min_, max_ = bounds
        std = epsilon / np.sqrt(3) * (max_ - min_)
        noise = nprng.normal(scale=std, size=x.shape)
        noise = noise.astype(x.dtype)
        return noise
