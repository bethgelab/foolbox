import numpy as np
from collections import Iterable

from scipy.ndimage.filters import gaussian_filter

from .base import Attack
from .base import call_decorator


class GaussianBlurAttack(Attack):
    """Blurs the input until it is misclassified."""

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 epsilons=1000):

        """Blurs the input until it is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if input is a `numpy.ndarray`, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        epsilons : int or Iterable[float]
            Either Iterable of standard deviations of the Gaussian blur
            or number of standard deviations between 0 and 1 that should
            be tried.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        x = a.unperturbed
        min_, max_ = a.bounds()
        axis = a.channel_axis(batch=False)
        hw = [x.shape[i] for i in range(x.ndim) if i != axis]
        h, w = hw
        size = max(h, w)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in epsilons:
            # epsilon = 1 will correspond to
            # sigma = size = max(width, height)
            sigmas = [epsilon * size] * 3
            sigmas[axis] = 0
            blurred = gaussian_filter(x, sigmas)
            blurred = np.clip(blurred, min_, max_)

            _, is_adversarial = a.forward_one(blurred)
            if is_adversarial:
                return
