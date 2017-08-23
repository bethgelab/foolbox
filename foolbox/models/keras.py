from __future__ import absolute_import
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

        if predicts == 'probs':
            predicts = 'probabilities'
        assert predicts in ['probabilities', 'logits']

        self.images_input = model.input
        self.label_input = K.placeholder(shape=(1,))

        self.predictions = model.output

        if predicts == 'probabilities':
            self.predictions_are_logits = False
        elif predicts == 'logits':
            self.predictions_are_logits = True

        shape = K.int_shape(self.predictions)
        _, num_classes = shape
        assert num_classes is not None

        self._num_classes = num_classes

        # create caches for loss
        self._loss_cache = {}
        self._loss_fn_cache = {}
        self._batch_pred_fn_cache = {}
        self._pred_grad_fn_cache = {}

    def _loss(loss):
        try:
            return self._loss_cache[loss]
        except KeyError:
            if hasattr(loss, '__call__'):
                return loss(self.predictions, self.label_input)
            elif loss in [None, 'logits']:
                return -self.predictions[0, self.label_input]
            elif loss == 'crossentropy':
                loss = K.sparse_categorical_crossentropy(
                    self.predictions, self.label_input,
                    from_logits=self.predictions_are_logits)

                # sparse_categorical_crossentropy returns 1-dim tensor,
                # gradients wants 0-dim tensor (for some backends)
                self._loss_cache[loss] = K.squeeze(loss, axis=0)
            elif loss == 'carlini':
                loss = K.max(self.predictions[0, :]) \
                       - self.predictions[0, self.label_input]
                return K.relu(loss)
            else:
                raise NotImplementedError('The loss {} is currently not \
                        implemented for this framework.'.format(loss))

        grads = K.gradients(loss, images_input)
        if K.backend() == 'tensorflow':
            # tensorflow backend returns a list with the gradient
            # as the only element, even if loss is a single scalar
            # tensor;
            # theano always returns the gradient itself (and requires
            # that loss is a single scalar tensor)
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]
        elif K.backend() == 'cntk':  # pragma: no cover
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]
            grad = K.reshape(grad, (1,) + grad.shape)
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

class KerasGradientCache(object):
    """ Cache of previously computed gradients """

    def __init__(self):
        self.gradients = {}

    def __call__(self, loss):
        try:
            return self.gradients[loss]
        except KeyError:
            # compute gradient
            pass
