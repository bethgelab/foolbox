from __future__ import absolute_import
import numpy as np
from .base import DifferentiableModel


class TheanoModel(DifferentiableModel):
    """Creates a :class:`Model` instance from existing `Theano` tensors.

    Parameters
    ----------
    images : `theano.tensor`
        The input to the model.
    logits : `theano.tensor`
        The predictions of the model, before the softmax.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    num_classes : int
        Number of classes for which the model will output predictions.
    channel_axis : int
        The index of the axis that represents color channels.

    """

    def __init__(
            self,
            images,
            logits,
            bounds,
            num_classes,
            channel_axis=1):

        super(TheanoModel, self).__init__(bounds=bounds,
                                          channel_axis=channel_axis)

        self._num_classes = num_classes

        # delay import until class is instantiated
        import theano as th
        import theano.tensor as T

        probs = T.nnet.nnet.softmax(logits)

        labels = T.ivector('labels')
        loss = T.nnet.nnet.categorical_crossentropy(
            probs, labels)
        gradient = th.gradient.grad(loss[0], images)

        self._batch_prediction_fn = th.function([images], logits)
        self._predictions_and_gradient_fn = th.function(
            [images, labels], [logits, gradient])
        self._gradient_fn = th.function([images, labels], gradient)

    def batch_predictions(self, images):
        predictions = self._batch_prediction_fn(images)
        assert predictions.shape == (images.shape[0], self.num_classes())
        return predictions

    def predictions_and_gradient(self, image, label):
        label = np.array(label, dtype=np.int32)
        predictions, gradient = self._predictions_and_gradient_fn(
            image[np.newaxis], label[np.newaxis])
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == image.shape
        return predictions, gradient

    def gradient(self, image, label):
        label = np.array(label, dtype=np.int32)
        gradient = self._gradient_fn(image[np.newaxis], label[np.newaxis])
        gradient = np.squeeze(gradient, axis=0)
        assert gradient.shape == image.shape
        return gradient

    def num_classes(self):
        return self._num_classes
