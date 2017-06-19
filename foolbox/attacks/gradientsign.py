from __future__ import division
import numpy as np
from collections import Iterable

from .base import Attack


class GradientSignAttack(Attack):
    """Adds the sign of the gradient to the image, gradually increasing
    the magnitude until the image is misclassified.

    Does not do anything if the model does not have a gradient.

    """

    def _apply(self, a, epsilons=1000):
        if not a.has_gradient():
            return

        image = a.original_image
        min_, max_ = a.bounds()
        gradient = a.gradient()
        gradient_sign = np.sign(gradient) * (max_ - min_)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            perturbed = image + gradient_sign * epsilon
            perturbed = np.clip(perturbed, min_, max_)

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                # TODO: if first epsilon, repeat with smaller epsilons
                return


FGSM = GradientSignAttack


class IterativeGradientSignAttack(Attack):
    """Like GradientSignAttack but with several steps for each epsilon.

    """

    def _apply(self, a, epsilons=100, steps=10):
        if not a.has_gradient():
            return

        image = a.original_image
        min_, max_ = a.bounds()

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1 / steps, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            perturbed = image

            for _ in range(steps):
                gradient = a.gradient(perturbed)
                gradient_sign = np.sign(gradient) * (max_ - min_)

                perturbed = image + gradient_sign * epsilon
                perturbed = np.clip(perturbed, min_, max_)

                a.predictions(perturbed)
                # we don't return early if an adversarial was found
                # because there might be a different epsilon
                # and/or step that results in a better adversarial
