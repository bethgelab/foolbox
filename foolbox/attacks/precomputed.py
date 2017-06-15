import numpy as np

from .base import Attack


class PrecomputedImagesAttack(Attack):
    """Attacks a model using precomputed adversarial candidates.

    Parameters
    ----------
    input_images : `numpy.ndarray`
        The original images that will be expected by this attack.
    output_images : `numpy.ndarray`
        The adversarial candidates corresponding to the input_images.
    """

    def __init__(self, input_images, output_images, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def _apply(self, a):
        image = a.original_image
        adversarial = self._get_output(a, image)
        a.predictions(adversarial)
