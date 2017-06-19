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
    predicts : str
        Specifies whether the `Keras` model predicts logits or probabilities.
        Logits are preferred, but probabilities are the default.
    preprocess_fn : function
        Will be called with the images before model predictions are calculated.

    """

    def __init__(
            self,
            model,
            bounds,
            channel_axis=3,
            predicts='probabilities',
            preprocess_fn=None):

        super(KerasModel, self).__init__(bounds=bounds,
                                         channel_axis=channel_axis)

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

        shape = predictions.get_shape().as_list()
        _, num_classes = shape
        assert num_classes is not None

        self._num_classes = num_classes

        loss = K.sparse_categorical_crossentropy(
            predictions, label_input, from_logits=predictions_are_logits)

        grads = K.gradients(loss, images_input)
        grad = grads[0]

        self._batch_pred_fn = K.function(
            [images_input], [predictions])
        self._pred_grad_fn = K.function(
            [images_input, label_input],
            [predictions, grad])

        self._predictions_are_logits = predictions_are_logits

        if preprocess_fn is not None:
            self.preprocessing_fn = lambda x: preprocess_fn(x.copy())
        else:
            self.preprocessing_fn = lambda x: x

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
        predictions = self._batch_pred_fn([self.preprocessing_fn(images)])
        assert len(predictions) == 1
        predictions = predictions[0]
        assert predictions.shape == (images.shape[0], self.num_classes())
        predictions = self._as_logits(predictions)
        return predictions

    def predictions_and_gradient(self, image, label):
        predictions, gradient = self._pred_grad_fn([
            self.preprocessing_fn(image[np.newaxis]),
            np.array([label])])
        predictions = np.squeeze(predictions, axis=0)
        predictions = self._as_logits(predictions)
        gradient = np.squeeze(gradient, axis=0)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == image.shape
        return predictions, gradient
