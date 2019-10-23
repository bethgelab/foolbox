import logging
import warnings
from collections import Iterable

import numpy as np

from .base import Attack
from .base import generator_decorator
from .. import nprng


class BlendedUniformNoiseAttack(Attack):
    """Blends the input with a uniform noise input until it is misclassified.

    """

    @generator_decorator
    def as_generator(self, a, epsilons=1000, max_directions=1000):

        """Blends the input with a uniform noise input until it is misclassified.

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
            Either Iterable of blending steps or number of blending steps
            between 0 and 1 that should be tried.
        max_directions : int
            Maximum number of random inputs to try.

        """

        x = a.unperturbed
        min_, max_ = a.bounds()

        if a.perturbed is not None:  # pragma: no cover
            warnings.warn(
                "BlendedUniformNoiseAttack started with"
                " previously found adversarial."
            )

        for j in range(max_directions):
            # random noise inputs tend to be classified into the same class,
            # so we might need to make very many draws if the original class
            # is that one
            random = nprng.uniform(min_, max_, size=x.shape).astype(x.dtype)
            _, is_adversarial = yield from a.forward_one(random)
            if is_adversarial:
                logging.info(
                    "Found adversarial input after {} " "attempts".format(j + 1)
                )
                break
        else:
            # never breaked
            warnings.warn(
                "BlendedUniformNoiseAttack failed to draw a"
                " random input that is adversarial."
            )

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            perturbed = (1 - epsilon) * x + epsilon * random
            # due to limited floating point precision,
            # clipping can be required
            if not a.in_bounds(perturbed):  # pragma: no cover
                np.clip(perturbed, min_, max_, out=perturbed)

            _, is_adversarial = yield from a.forward_one(perturbed)
            if is_adversarial:
                return
