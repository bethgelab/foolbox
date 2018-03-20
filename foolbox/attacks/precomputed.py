import numpy as np

from .base import Attack
from .base import call_decorator


class PrecomputedImagesAttack(Attack):
    """Attacks a model using precomputed adversarial candidates.

    Parameters
    ----------
    input_images : `numpy.ndarray`
        The original images that will be expected by this attack.
    output_images : `numpy.ndarray`
        The adversarial candidates corresponding to the input_images.
    *args : positional args
        Poistional args passed to the `Attack` base class.
    **kwargs : keyword args
        Keyword args passed to the `Attack` base class.
    """

    def __init__(self, input_images, output_images, *args, **kwargs):
        super(PrecomputedImagesAttack, self).__init__(*args, **kwargs)

        assert input_images.shape == output_images.shape

        self._input_images = input_images
        self._output_images = output_images

    def _get_output(self, a, image):
        """ Looks up the precomputed adversarial image for a given image.

        """
        sd = np.square(self._input_images - image)
        mses = np.mean(sd, axis=tuple(range(1, sd.ndim)))
        index = np.argmin(mses)

        # if we run into numerical problems with this approach, we might
        # need to add a very tiny threshold here
        if mses[index] > 0:
            raise ValueError('No precomputed output image for this image')
        return self._output_images[index]

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True):
        """Attacks a model using precomputed adversarial candidates.

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

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        image = a.original_image
        adversarial = self._get_output(a, image)
        a.predictions(adversarial)
