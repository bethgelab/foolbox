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
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.

    """

    def __init__(
            self,
            images,
            logits,
            bounds,
            num_classes,
            channel_axis=1,
            preprocessing=[0, 1]):

        super(TheanoModel, self).__init__(bounds=bounds,
                                          channel_axis=channel_axis,
                                          preprocessing=preprocessing)

        self._num_classes = num_classes

        # delay import until class is instantiated
        import theano as th
        import theano.tensor as T

        probs = T.nnet.nnet.softmax(logits)

        labels = T.ivector('labels')
        loss = T.nnet.nnet.categorical_crossentropy(
            probs, labels)
        gradient = th.gradient.grad(loss[0], images)

        bw_gradient_pre = T.fmatrix('bw_gradient_pre')
        bw_loss = (logits * bw_gradient_pre).sum()
        bw_gradient = th.gradient.grad(bw_loss, images)

        self._batch_prediction_fn = th.function([images], logits)
        self._predictions_and_gradient_fn = th.function(
            [images, labels], [logits, gradient])
        self._gradient_fn = th.function([images, labels], gradient)
        self._loss_fn = th.function([images, labels], loss)
        self._bw_gradient_fn = th.function(
            [bw_gradient_pre, images], bw_gradient)

    def batch_predictions(self, images):
        images = self._process_input(images)
        predictions = self._batch_prediction_fn(images)
        assert predictions.shape == (images.shape[0], self.num_classes())
        return predictions

    def predictions_and_gradient(self, image, label):
        image = self._process_input(image)
        label = np.array(label, dtype=np.int32)
        predictions, gradient = self._predictions_and_gradient_fn(
            image[np.newaxis], label[np.newaxis])
        predictions = np.squeeze(predictions, axis=0)
        gradient = self._process_gradient(gradient)
        gradient = np.squeeze(gradient, axis=0)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == image.shape
        assert gradient.dtype == image.dtype
        return predictions, gradient

    def gradient(self, image, label):
        image = self._process_input(image)
        label = np.array(label, dtype=np.int32)
        gradient = self._gradient_fn(image[np.newaxis], label[np.newaxis])
        gradient = self._process_gradient(gradient)
        gradient = np.squeeze(gradient, axis=0)
        assert gradient.shape == image.shape
        assert gradient.dtype == image.dtype
        return gradient

    def num_classes(self):
        return self._num_classes

    def backward(self, gradient, image):
        assert gradient.ndim == 1
        image = self._process_input(image)
        gradient = self._bw_gradient_fn(
            gradient[np.newaxis], image[np.newaxis])
        gradient = self._process_gradient(gradient)
        gradient = np.squeeze(gradient, axis=0)
        assert gradient.shape == image.shape
        assert gradient.dtype == image.dtype
        return gradient
