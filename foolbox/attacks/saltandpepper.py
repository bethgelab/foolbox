import numpy as np

from .base import Attack


class SaltAndPepperNoiseAttack(Attack):
    """Increases the amount of salt and pepper noise until the
    image is misclassified.

    """

    def _apply(self, a, *, epsilons=100, repetitions=10):
        image = a.original_image()
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

                u = np.random.uniform(size=shape)
                u = u.repeat(channels, axis=axis)

                salt = (u >= 1 - p / 2).astype(image.dtype) * r
                pepper = -(u < p / 2).astype(image.dtype) * r

                perturbed = image + salt + pepper
                perturbed = np.clip(perturbed, min_, max_)

                if a.normalized_distance(perturbed) >= a.best_distance():
                    continue

                _, is_adversarial = a.predictions(perturbed)
                if is_adversarial:
                    # higher epsilon usually means larger perturbation, but
                    # this relationship is not strictly monotonic, so we set
                    # the new limit a bit higher than the best one so far
                    max_epsilon = epsilon * 1.2
                    break
