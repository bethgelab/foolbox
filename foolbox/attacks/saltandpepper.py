import numpy as np

from .base import Attack
from .base import generator_decorator
from .. import nprng


class SaltAndPepperNoiseAttack(Attack):
    """Increases the amount of salt and pepper noise until the input is misclassified.

    """

    @generator_decorator
    def as_generator(self, a, epsilons=100, repetitions=10):

        """Increases the amount of salt and pepper noise until the input is misclassified.

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

        x = a.unperturbed
        min_, max_ = a.bounds()
        axis = a.channel_axis(batch=False)
        channels = x.shape[axis]
        shape = list(x.shape)
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

                salt = (u >= 1 - p / 2).astype(x.dtype) * r
                pepper = -(u < p / 2).astype(x.dtype) * r

                perturbed = x + salt + pepper
                perturbed = np.clip(perturbed, min_, max_)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    # higher epsilon usually means larger perturbation, but
                    # this relationship is not strictly monotonic, so we set
                    # the new limit a bit higher than the best one so far
                    # but not larger than 1
                    max_epsilon = min(1, epsilon * 1.2)
                    break
