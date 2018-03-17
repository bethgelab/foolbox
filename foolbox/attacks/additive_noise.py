from abc import abstractmethod
from collections import Iterable

import numpy as np

from .base import Attack
from .base import call_decorator


class AdditiveNoiseAttack(Attack):
    """Base class for attacks that add random noise to an image.

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True, epsilons=1000):
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        image = a.original_image
        bounds = a.bounds()
        min_, max_ = bounds

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            noise = self._sample_noise(epsilon, image, bounds)
            perturbed = image + epsilon * noise
            perturbed = np.clip(perturbed, min_, max_)

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                return

    @abstractmethod
    def _sample_noise(self):
        raise NotImplementedError


class AdditiveUniformNoiseAttack(AdditiveNoiseAttack):
    """Adds uniform noise to the image, gradually increasing
    the standard deviation until the image is misclassified.

    """

    def _sample_noise(self, epsilon, image, bounds):
        min_, max_ = bounds
        w = epsilon * (max_ - min_)
        noise = np.random.uniform(-w, w, size=image.shape)
        noise = noise.astype(image.dtype)
        return noise


class AdditiveGaussianNoiseAttack(AdditiveNoiseAttack):
    """Adds Gaussian noise to the image, gradually increasing
    the standard deviation until the image is misclassified.

    """

    def _sample_noise(self, epsilon, image, bounds):
        min_, max_ = bounds
        std = epsilon / np.sqrt(3) * (max_ - min_)
        noise = np.random.normal(scale=std, size=image.shape)
        noise = noise.astype(image.dtype)
        return noise
