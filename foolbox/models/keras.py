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
        self.label_input = K.placeholder(shape=(1,), dtype='int32')

        # predictions should always be logits
        self.output = model.output
        if predicts == 'probabilities':
            self.predictions_are_logits = False
            self.logits = self._as_logits(self.output)
        elif predicts == 'logits':
            self.predictions_are_logits = True
            self.logits = self.output

        shape = K.int_shape(self.logits)
        _, num_classes = shape
        assert num_classes is not None

        self._num_classes = num_classes

        self._batch_pred_fn = K.function(
            [self.images_input], [self.logits])

        # create caches for loss
        self._loss_cache = {}
        self._grad_cache = {}
        self._loss_fn_cache = {}
        self._pred_grad_fn_cache = {}

    def _loss(self, loss):
        from keras import backend as K
        try:
            return self._loss_cache[loss]
        except KeyError:
            if hasattr(loss, '__call__'):
                return loss(self.logits, self.label_input)
            elif loss in [None, 'logits']:
                sym_loss = -K.sum(self.logits[:, self.label_input[0]])
            elif loss == 'crossentropy':
                sym_loss = K.sparse_categorical_crossentropy(
                    self.label_input, self.output,
                    from_logits=self.predictions_are_logits)

                # sparse_categorical_crossentropy returns 1-dim tensor,
                # gradients wants 0-dim tensor (for some backends)
                sym_loss = K.squeeze(sym_loss, axis=0)
            elif loss == 'carlini':
                sym_loss = K.max(K.sum(self.logits, axis=0)) \
                       - K.sum(self.logits[:, self.label_input[0]])
                sym_loss = K.relu(sym_loss)
            else:
                raise NotImplementedError('The loss {} is currently not \
                        implemented for this framework.'.format(loss))

            self._loss_cache[loss] = sym_loss
            return sym_loss

    def _gradient(self, loss):
        from keras import backend as K
        try:
            return self._grad_cache[loss]
        except KeyError:
            sym_loss = self._loss(loss)
            grads = K.gradients(sym_loss, self.images_input)
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

            self._grad_cache[loss] = grad
            return grad

    def _pred_grad_fn(self, input, label, loss=None):
        from keras import backend as K
        input = self._process_input(input)
        try:
            return self._pred_grad_fn_cache[loss]([input, label])
        except KeyError:
            sym_grad = self._gradient(loss)

            self._pred_grad_fn_cache[loss] =  K.function(
                [self.images_input, self.label_input],
                [self.logits, sym_grad])

            return self._pred_grad_fn_cache[loss]([input, label])

    def _loss_fn(self, input, label, loss=None):
        from keras import backend as K
        inputs = self._process_input(input)[None]
        label = [label]
        try:
            return self._loss_fn_cache[loss]([inputs, label])[0]
        except KeyError:
            sym_loss = self._loss(loss)

            self._loss_fn_cache[loss] =  K.function(
                [self.images_input, self.label_input],
                [sym_loss])

            return self._loss_fn_cache[loss]([inputs, label])[0]

    def _as_logits(self, predictions):
        from keras import backend as K
        if self.predictions_are_logits:
            return predictions
        eps = 10e-8
        predictions = K.clip(predictions, eps, 1 - eps)
        predictions = K.log(predictions)
        return predictions

    def num_classes(self):
        return self._num_classes

    def batch_predictions(self, images):
        predictions = self._batch_pred_fn([self._process_input(images)])
        assert len(predictions) == 1
        predictions = predictions[0]
        assert predictions.shape == (images.shape[0], self.num_classes())
        return predictions

    def predictions_and_gradient(self, image, label, loss=None):
        predictions, gradient = self._pred_grad_fn(
            self._process_input(image[np.newaxis]),
            np.array([label]),
            loss=loss)
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(gradient)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == image.shape
        return predictions, gradient
