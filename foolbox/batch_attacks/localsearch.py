from .base import BatchAttack
from .base import generator_decorator
from .. import nprng


class SinglePixelAttack(BatchAttack):
    """Perturbs just a single pixel and sets it to the min or max."""

    @generator_decorator
    def as_generator(self, a, max_pixels=1000):

        """Perturbs just a single pixel and sets it to the min or max.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified input. If it is a
            numpy array, label must be passed as well. If it is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original input. Must be passed
            if input is a numpy array, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        max_pixels : int
            Maximum number of pixels to try.

        """

        channel_axis = a.channel_axis(batch=False)
        axes = [i for i in range(a.unperturbed.ndim) if i != channel_axis]
        assert len(axes) == 2
        h = a.unperturbed.shape[axes[0]]
        w = a.unperturbed.shape[axes[1]]

        min_, max_ = a.bounds()

        pixels = nprng.permutation(h * w)
        pixels = pixels[:max_pixels]
        for i, pixel in enumerate(pixels):
            x = pixel % w
            y = pixel // w

            location = [x, y]
            location.insert(channel_axis, slice(None))
            location = tuple(location)

            for value in [min_, max_]:
                perturbed = a.unperturbed.copy()
                perturbed[location] = value

                _, is_adv = yield from a.forward_one(perturbed)
                if is_adv:
                    return
