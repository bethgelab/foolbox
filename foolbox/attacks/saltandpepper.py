import numpy as np

from .base import Attack
from .base import call_decorator
from .. import nprng


class SaltAndPepperNoiseAttack(Attack):
    """Increases the amount of salt and pepper noise until the
    image is misclassified.

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 epsilons=100, repetitions=10):

        """Increases the amount of salt and pepper noise until the
        image is misclassified.

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
        epsilons : int
            Number of steps to try between probability 0 and 1.
        repetitions : int
            Specifies how often the attack will be repeated.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()
        axis = a.channel_axis(batch=False)
        channels = image.shape[axis]
        shape = list(image.shape)
        shape[axis] = 1
        r = max_ - min_
        pixels = np.prod(shape)

        epsilons = min(epsilons, pixels)
        max_epsilon = 1

        for _ in range(repetitions):
            for epsilon in np.linspace(0, max_epsilon, num=epsilons + 1)[1:]:
                p = epsilon

                u = nprng.uniform(size=shape)
                u = u.repeat(channels, axis=axis)

                salt = (u >= 1 - p / 2).astype(image.dtype) * r
                pepper = -(u < p / 2).astype(image.dtype) * r

                perturbed = image + salt + pepper
                perturbed = np.clip(perturbed, min_, max_)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = a.predictions(perturbed)
                if is_adversarial:
                    # higher epsilon usually means larger perturbation, but
                    # this relationship is not strictly monotonic, so we set
                    # the new limit a bit higher than the best one so far
                    max_epsilon = epsilon * 1.2
                    break
