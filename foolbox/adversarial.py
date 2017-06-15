"""
Provides a class that represents an adversarial example.

"""

import numpy as np
import numbers

from .distances import MSE


class Adversarial(object):
    """Defines an adversarial that should be found and stores the result.

    The :class:`Adversarial` class represents a single adversarial example
    for a given model, criterion and reference image. It can be passed to
    an adversarial attack to find the actual adversarial.

    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
    criterion : a :class:`Criterion` instance
        The criterion that determines which images are adversarial.
    original_image : a :class:`numpy.ndarray`
        The original image to which the adversarial image should
        be as close as possible.
    original_class : int
        The ground-truth label of the original image.
    distance : a :class:`Distance` class
        The measure used to quantify similarity between images.

    """
    def __init__(
            self,
            model,
            criterion,
            original_image,
            original_class,
            *,
            distance=MSE):

        self.__model = model
        self.__criterion = criterion
        self.__original_image = original_image
        self.__original_class = original_class
        self.__distance = distance

        self.__best_adversarial = None
        self.__best_distance = distance(value=np.inf)

        self._total_prediction_calls = 0
        self._total_gradient_calls = 0

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        # check if the original image is already adversarial
        self.predictions(original_image)

    @property
    def image(self):
        return self.__best_adversarial

    @property
    def distance(self):
        return self.__best_distance

    @property
    def original_image(self):
        return self.__original_image

    @property
    def original_class(self):
        return self.__original_class

    def normalized_distance(self, image):
        """Calculates the distance of a given image to the
        original image.

        Parameters
        ----------
        image : `numpy.ndarray`
            The image that should be compared to the original image.

        Returns
        -------
        :class:`Distance`
            The distance between the given image and the original image.

        """
        min_, max_ = self.bounds()
        r = max_ - min_
        return self.__distance(self.__original_image / r, image / r)

    def __new_adversarial(self, image):
        distance = self.normalized_distance(image)
        if self.__best_distance > distance:
            # new best adversarial

            self.__best_adversarial = image
            self.__best_distance = distance

            self._best_prediction_calls = self._total_prediction_calls
            self._best_gradient_calls = self._total_gradient_calls

            return True
        return False

    def __is_adversarial(self, image, predictions):
        """Interface to criterion.is_adverarial that calls
        __new_adversarial if necessary.

        Parameters
        ----------
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some image.
        label : int
            The label of the unperturbed reference image.

        """
        is_adversarial = self.__criterion.is_adversarial(
            predictions, self.__original_class)
        if is_adversarial:
            self.__new_adversarial(image)
        assert isinstance(is_adversarial, bool) or \
            isinstance(is_adversarial, np.bool_)
        return is_adversarial

    def target_class(self):
        """Interface to criterion.target_class for attacks.

        """
        try:
            target_class = self.__criterion.target_class()
        except AttributeError:
            target_class = None
        return target_class

    def num_classes(self):
        n = self.__model.num_classes()
        assert isinstance(n, numbers.Number)
        return n

    def bounds(self):
        min_, max_ = self.__model.bounds()
        assert isinstance(min_, numbers.Number)
        assert isinstance(max_, numbers.Number)
        assert min_ < max_
        return min_, max_

    def in_bounds(self, input_):
        min_, max_ = self.bounds()
        return min_ <= input_.min() and input_.max() <= max_

    def channel_axis(self, *, batch):
        """Interface to model.channel_axis for attacks.

        Parameters
        ----------
        batch : bool
            Controls whether the index of the axis for a batch of images
            (4 dimensions) or a single image (3 dimensions) should be returned.

        """
        axis = self.__model.channel_axis()
        if not batch:
            axis = axis - 1
        return axis

    def has_gradient(self):
        """Returns true if _backward and _forward_backward can be called
        by an attack, False otherwise.

        """
        try:
            self.__model.gradient
            self.__model.predictions_and_gradient
        except AttributeError:
            return False
        else:
            return True

    def predictions(self, image, strict=True):
        """Interface to model.predictions for attacks.

        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert not strict or self.in_bounds(image)

        self._total_prediction_calls += 1
        predictions = self.__model.predictions(image)
        is_adversarial = self.__is_adversarial(image, predictions)

        assert predictions.ndim == 1
        return predictions, is_adversarial

    def batch_predictions(self, images, *, increasing=False, strict=True):
        """Interface to model.batch_predictions for attacks.

        Parameters
        ----------
        images : `numpy.ndarray`
            Batch of images with shape (batch size, height, width, channels).
        increasing : bool
            Whether the images can be assumed to have increasing
            distance from the original image.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert not strict or self.in_bounds(images)

        self._total_prediction_calls += len(images)
        predictions = self.__model.batch_predictions(images)

        assert predictions.ndim == 2
        assert predictions.shape[0] == images.shape[0]

        adversarials = []
        for i in range(len(predictions)):
            is_adversarial = self.__is_adversarial(images[i], predictions[i])
            if is_adversarial and increasing:
                return predictions, is_adversarial, i
            adversarials.append(is_adversarial)

        if increasing:
            # no adversarial found
            return predictions, False, None

        is_adversarial = np.array(adversarials)
        assert is_adversarial.ndim == 1
        assert is_adversarial.shape[0] == images.shape[0]

        return predictions, is_adversarial

    def gradient(self, image=None, label=None, strict=True):
        """Interface to model.gradient for attacks.

        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
            Defaults to the original image.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if image is None:
            image = self.__original_image
        if label is None:
            label = self.__original_class

        assert not strict or self.in_bounds(image)

        self._total_gradient_calls += 1
        gradient = self.__model.gradient(image, label)

        assert gradient.shape == image.shape
        return gradient

    def predictions_and_gradient(self, image=None, label=None, strict=True):
        """Interface to model.predictions_and_gradient for attacks.

        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
            Defaults to the original image.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if image is None:
            image = self.__original_image
        if label is None:
            label = self.__original_class

        assert not strict or self.in_bounds(image)

        self._total_prediction_calls += 1
        self._total_gradient_calls += 1
        predictions, gradient = self.__model.predictions_and_gradient(image, label)  # noqa: E501
        is_adversarial = self.__is_adversarial(image, predictions)

        assert predictions.ndim == 1
        assert gradient.shape == image.shape
        return predictions, gradient, is_adversarial
