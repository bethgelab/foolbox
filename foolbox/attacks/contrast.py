import numpy as np
from collections import Iterable

from .base import Attack
from .base import call_decorator


class ContrastReductionAttack(Attack):
    """Reduces the contrast of the image until it is misclassified.

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 epsilons=1000):
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()
        target = (max_ + min_) / 2

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            perturbed = (1 - epsilon) * image + epsilon * target

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                return
