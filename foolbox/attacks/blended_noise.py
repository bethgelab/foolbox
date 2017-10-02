import warnings
from collections import Iterable

import numpy as np

from .base import Attack


class BlendedUniformNoiseAttack(Attack):
    """Blends the image with a uniform noise image until it
    is misclassified.

    """

    def _apply(self, a, epsilons=1000):
        image = a.original_image
        min_, max_ = a.bounds()

        if a.image is not None:  # pramga: no cover
            warnings.warn('BlendedUniformNoiseAttack started with previously found adversarial.')  # noqa: E501

        random_image = np.random.uniform(
            min_, max_, size=image.shape).astype(image.dtype)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            perturbed = (1 - epsilon) * image + epsilon * random_image
            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                return
