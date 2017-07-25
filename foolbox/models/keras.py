from __future__ import absolute_import
import warnings
import numpy as np

from .base import DifferentiableModel


class KerasModel(DifferentiableModel):
    """Creates a :class:`Model` instance from a `Keras` model.

    Parameters
    ----------
    model : `keras.models.Model`
        The `Keras` model that should be attacked.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    predicts : str
        Specifies whether the `Keras` model predicts logits or probabilities.
        Logits are preferred, but probabilities are the default.

    """

    def __init__(
            self,
            model,
            bounds,
            channel_axis=3,
            preprocessing=(0, 1),
            predicts='probabilities'):

        super(KerasModel, self).__init__(bounds=bounds,
                                         channel_axis=channel_axis,
                                         preprocessing=preprocessing)

        from keras import backend as K

        if K.backend() != 'tensorflow':  # pragma: no cover
            warnings.warn('Your keras backend might not be supported.')

        if predicts == 'probs':
            predicts = 'probabilities'
        assert predicts in ['probabilities', 'logits']

        images_input = model.input
        label_input = K.placeholder(shape=(1,))

        predictions = model.output

        if predicts == 'probabilities':
            predictions_are_logits = False
        elif predicts == 'logits':
            predictions_are_logits = True

        shape = K.int_shape(predictions)
        _, num_classes = shape
        assert num_classes is not None

        self._num_classes = num_classes

        loss = K.sparse_categorical_crossentropy(
            predictions, label_input, from_logits=predictions_are_logits)

        # sparse_categorical_crossentropy returns 1-dim tensor,
        # gradients wants 0-dim tensor (for some backends)
        loss = K.squeeze(loss, axis=0)

        grads = K.gradients(loss, images_input)
        if K.backend() == 'tensorflow':
            # tensorflow backend returns a list with the gradient
            # as the only element, even if loss is a single scalar
            # tensor;
            # theano always returns the gradient itself (and requires
            # that loss is a single scalar tensor)
            assert isinstance(grads, list)
            grad = grads[0]
        else:
            assert not isinstance(grads, list)
            grad = grads

        self._loss_fn = K.function(
            [images_input, label_input],
            [loss])
        self._batch_pred_fn = K.function(
            [images_input], [predictions])
        self._pred_grad_fn = K.function(
            [images_input, label_input],
            [predictions, grad])

        self._predictions_are_logits = predictions_are_logits

    def _as_logits(self, predictions):
        assert predictions.ndim in [1, 2]
        if self._predictions_are_logits:
            return predictions
        eps = 10e-8
        predictions = np.clip(predictions, eps, 1 - eps)
        predictions = np.log(predictions)
        return predictions

    def num_classes(self):
        return self._num_classes

    def batch_predictions(self, images):
        predictions = self._batch_pred_fn([self._process_input(images)])
        assert len(predictions) == 1
        predictions = predictions[0]
        assert predictions.shape == (images.shape[0], self.num_classes())
        predictions = self._as_logits(predictions)
        return predictions

    def predictions_and_gradient(self, image, label):
        predictions, gradient = self._pred_grad_fn([
            self._process_input(image[np.newaxis]),
            np.array([label])])
        predictions = np.squeeze(predictions, axis=0)
        predictions = self._as_logits(predictions)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(gradient)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == image.shape
        return predictions, gradient
