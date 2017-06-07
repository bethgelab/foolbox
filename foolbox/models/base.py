import numpy as np
from abc import ABC, abstractmethod


class Model(ABC):
    """Base class to provide attacks with a unified interface to models.

    The :class:`Model` class represents a model and provides a
    unified interface to its predictions. Subclasses must implement
    batch_predictions and num_classes.

    :class:`Model` instances can be used as context managers and subclasses
    can require this to allocate and release resources.

    Parameters
    ----------
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.

    """

    def __init__(self, *, bounds, channel_axis):
        assert len(bounds) == 2
        self._bounds = bounds
        assert channel_axis in [0, 1, 2, 3]
        self._channel_axis = channel_axis

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def bounds(self):
        return self._bounds

    def channel_axis(self):
        return self._channel_axis

    @abstractmethod
    def batch_predictions(self, images):
        """Calculates predictions for a batch of images.

        Parameters
        ----------
        images : `numpy.ndarray`
            Batch of images with shape (batch size, height, width, channels).

        Returns
        -------
        `numpy.ndarray`
            Predictions (logits, i.e. beforre the softmax) with shape
            (batch size, number of classes).

        See Also
        --------
        :meth:`prediction`

        """
        raise NotImplementedError

    def predictions(self, image):
        """Convenience method that calculates predictions for a single image.

        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).

        Returns
        -------
        `numpy.ndarray`
            Vector of predictions (logits, i.e. beforre the softmax) with
            shape (number of classes,).

        See Also
        --------
        :meth:`batch_predictions`

        """
        return np.squeeze(self.batch_predictions(image[np.newaxis]), axis=0)

    @abstractmethod
    def num_classes(self):
        """Determines the number of classes.

        Returns
        -------
        int
            The number of classes for which the model creates predictions.

        """
        raise NotImplementedError


class DifferentiableModel(Model):
    """Base class for differentiable models that provide gradients.

    The :class:`DifferentiableModel` class can be used as a base
    class for models that provide gradients. Subclasses must implement
    predictions_and_gradient.

    A model should be considered differentiable based on whether it
    provides a :meth:`predictions_and_gradient` method and a
    :meth:`gradient` method, not based on whether it subclasses
    :class:`DifferentiableModel`.

    A differentiable model does not necessarily provide reasonable
    values for the gradients, the gradient can be wrong. It only
    guarantees that the relevant methods can be called.

    """

    @abstractmethod
    def predictions_and_gradient(self, image, label):
        """Calculates predictions for an image and the gradient of
        the cross-entropy loss w.r.t. the image.

        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
        label : int
            Reference label used to calculate the gradient.

        Returns
        -------
        predictions : `numpy.ndarray`
            Vector of predictions (logits, i.e. beforre the softmax) with
            shape (number of classes,).
        gradient : `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the image. Will
            have the same shape as the image.

        See Also
        --------
        :meth:`gradient`

        """
        raise NotImplementedError

    def gradient(self, image, label):
        """Calculates the gradient of the cross-entropy loss w.r.t. the image.

        The default implementation calls predictions_and_gradient.
        Subclasses can provide more efficient implementations that
        only calculate the gradient.

        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
        label : int
            Reference label used to calculate the gradient.

        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the image. Will
            have the same shape as the image.

        See Also
        --------
        :meth:`gradient`

        """
        _, gradient = self.predictions_and_gradient(image, label)
        return gradient
