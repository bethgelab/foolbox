from __future__ import absolute_import

import numpy as np

from .base import DifferentiableModel


class TensorFlowEagerModel(DifferentiableModel):
    """Creates a :class:`Model` instance from a `TensorFlow` model using
    eager execution.

    Parameters
    ----------
    model : a TensorFlow eager model
        The TensorFlow eager model that should be attacked. It will be called
        with input tensors and should return logits.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    num_classes : int
        If None, will try to infer it from the model's output shape.
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    """

    def __init__(
            self,
            model,
            bounds,
            num_classes=None,
            channel_axis=3,
            preprocessing=(0, 1)):

        # delay import until class is instantiated
        import tensorflow as tf
        assert tf.executing_eagerly()

        super(TensorFlowEagerModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self._model = model

        if num_classes is None:
            try:
                num_classes = model.output_shape[-1]
            except AttributeError:
                raise ValueError(
                    'Please specify num_classes manually or '
                    'provide a model with an output_shape attribute')

        self._num_classes = num_classes

    def batch_predictions(self, images):
        import tensorflow as tf
        images, _ = self._process_input(images)
        n = len(images)
        images = tf.constant(images)

        predictions = self._model(images)
        predictions = predictions.numpy()
        assert predictions.ndim == 2
        assert predictions.shape == (n, self.num_classes())
        return predictions

    def num_classes(self):
        return self._num_classes

    def predictions_and_gradient(self, image, label):
        import tensorflow as tf
        input_shape = image.shape
        image, dpdx = self._process_input(image)
        images = image[np.newaxis]
        images = tf.constant(images)
        target = tf.constant([label])

        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = self._model(images)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target, logits=predictions)

        grad = tape.gradient(loss, images)

        predictions = predictions.numpy()
        predictions = np.squeeze(predictions, axis=0)
        assert predictions.ndim == 1
        assert predictions.shape == (self.num_classes(),)

        grad = grad.numpy()
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return predictions, grad

    def _loss_fn(self, image, label):
        import tensorflow as tf
        image, _ = self._process_input(image)
        images = image[np.newaxis]
        images = tf.constant(images)
        target = tf.constant([label])

        predictions = self._model(images)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=predictions)
        loss = loss.numpy()
        return loss

    def backward(self, gradient, image):
        import tensorflow as tf
        input_shape = image.shape
        image, dpdx = self._process_input(image)
        images = image[np.newaxis]
        images = tf.constant(images)
        assert gradient.ndim == 1
        gradient = gradient[np.newaxis]
        gradient = tf.constant(gradient)

        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = self._model(images)

        # backprop the given output gradient (the gradient of
        # some loss w.r.t. predictions) through the model
        # to get the gradient of that loss w.r.t. images
        grad = tape.gradient(predictions, images, gradient)

        grad = grad.numpy()
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return grad
