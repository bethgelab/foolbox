import numpy as np
from collections import Iterable

from .base import Attack
from .base import generator_decorator


class ContrastReductionAttack(Attack):
    """Reduces the contrast of the input until it is misclassified."""

    @generator_decorator
    def as_generator(self, a, epsilons=1000):

        """Reduces the contrast of the input until it is misclassified.

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
            Either Iterable of contrast levels or number of contrast
            levels between 1 and 0 that should be tried. Epsilons are
            one minus the contrast level.

        """

        x = a.unperturbed
        min_, max_ = a.bounds()
        target = (max_ + min_) / 2

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            perturbed = (1 - epsilon) * x + epsilon * target

            _, is_adversarial = yield from a.forward_one(perturbed)
            if is_adversarial:
                return
