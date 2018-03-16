import numpy as np
from collections import Iterable

from scipy.ndimage.filters import gaussian_filter

from .base import Attack
from .base import call_decorator


class GaussianBlurAttack(Attack):
    """Blurs the image until it is misclassified.

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
        axis = a.channel_axis(batch=False)
        hw = [image.shape[i] for i in range(image.ndim) if i != axis]
        h, w = hw
        size = max(h, w)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            # epsilon = 1 will correspond to
            # sigma = size = max(width, height)
            sigmas = [epsilon * size] * 3
            sigmas[axis] = 0
            blurred = gaussian_filter(image, sigmas)
            blurred = np.clip(blurred, min_, max_)

            _, is_adversarial = a.predictions(blurred)
            if is_adversarial:
                return
