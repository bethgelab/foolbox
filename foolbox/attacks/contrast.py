import numpy as np
from collections import Iterable

from .base import Attack


class ContrastReductionAttack(Attack):
    """Reduces the contrast of the image until it is misclassified.

    """

    def _apply(self, a, epsilons=1000):
        image = a.original_image
        min_, max_ = a.bounds()
        target = (max_ - min_) / 2

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            perturbed = (1 - epsilon) * image + epsilon * target

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                return
