import logging
import warnings
from collections import Iterable

import numpy as np

from .base import Attack
from .base import call_decorator
from .. import nprng


class BlendedUniformNoiseAttack(Attack):
    """Blends the image with a uniform noise image until it
    is misclassified.

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 epsilons=1000, max_directions=1000):

        """Blends the image with a uniform noise image until it
        is misclassified.

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
            Maximum number of random images to try.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()

        if a.image is not None:  # pragma: no cover
            warnings.warn('BlendedUniformNoiseAttack started with'
                          ' previously found adversarial.')

        for j in range(max_directions):
            # random noise images tend to be classified into the same class,
            # so we might need to make very many draws if the original class
            # is that one
            random_image = nprng.uniform(
                min_, max_, size=image.shape).astype(image.dtype)
            _, is_adversarial = a.predictions(random_image)
            if is_adversarial:
                logging.info('Found adversarial image after {} '
                             'attempts'.format(j + 1))
                break
        else:
            # never breaked
            warnings.warn('BlendedUniformNoiseAttack failed to draw a'
                          ' random image that is adversarial.')

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            perturbed = (1 - epsilon) * image + epsilon * random_image
            # due to limited floating point precision,
            # clipping can be required
            if not a.in_bounds(perturbed):  # pragma: no cover
                np.clip(perturbed, min_, max_, out=perturbed)

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                return
