from __future__ import absolute_import
import numpy as np
import logging

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
        import keras
        from pkg_resources import parse_version

        assert parse_version(keras.__version__) >= parse_version('2.0.7'), 'Keras version needs to be 2.0.7 or newer'  # noqa: E501

        if predicts == 'probs':
            predicts = 'probabilities'
        assert predicts in ['probabilities', 'logits']

        images_input = model.input
        label_input = K.placeholder(shape=(1,))

        predictions = model.output

        shape = K.int_shape(predictions)
        _, num_classes = shape
        assert num_classes is not None

        self._num_classes = num_classes

        if predicts == 'probabilities':
            if K.backend() == 'tensorflow':
                predictions, = predictions.op.inputs
                loss = K.sparse_categorical_crossentropy(
                    label_input, predictions, from_logits=True)
            else:
                logging.warning('relying on numerically unstable conversion'
                                ' from probabilities to softmax')
                loss = K.sparse_categorical_crossentropy(
                    label_input, predictions, from_logits=False)

                # transform the probability predictions into logits, so that
                # the rest of this code can assume predictions to be logits
                predictions = self._to_logits(predictions)
        elif predicts == 'logits':
            loss = K.sparse_categorical_crossentropy(
                label_input, predictions, from_logits=True)

        # sparse_categorical_crossentropy returns 1-dim tensor,
        # gradients wants 0-dim tensor (for some backends)
        loss = K.squeeze(loss, axis=0)
        grads = K.gradients(loss, images_input)

        grad_loss_output = K.placeholder(shape=(num_classes, 1))
        external_loss = K.dot(predictions, grad_loss_output)
        # remove batch dimension of predictions
        external_loss = K.squeeze(external_loss, axis=0)
        # remove singleton dimension of grad_loss_output
        external_loss = K.squeeze(external_loss, axis=0)

        grads_loss_input = K.gradients(external_loss, images_input)

        if K.backend() == 'tensorflow':
            # tensorflow backend returns a list with the gradient
            # as the only element, even if loss is a single scalar
            # tensor;
            # theano always returns the gradient itself (and requires
            # that loss is a single scalar tensor)
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]

            assert isinstance(grads_loss_input, list)
            assert len(grads_loss_input) == 1
            grad_loss_input = grads_loss_input[0]
        elif K.backend() == 'cntk':  # pragma: no cover
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]
            grad = K.reshape(grad, (1,) + grad.shape)

            assert isinstance(grads_loss_input, list)
            assert len(grads_loss_input) == 1
            grad_loss_input = grads_loss_input[0]
            grad_loss_input = K.reshape(grad_loss_input, (1,) + grad_loss_input.shape)  # noqa: E501
        else:
            assert not isinstance(grads, list)
            grad = grads

            grad_loss_input = grads_loss_input

        self._loss_fn = K.function(
            [images_input, label_input],
            [loss])
        self._batch_pred_fn = K.function(
            [images_input], [predictions])
        self._pred_grad_fn = K.function(
            [images_input, label_input],
            [predictions, grad])
        self._bw_grad_fn = K.function(
            [grad_loss_output, images_input],
            [grad_loss_input])

    def _to_logits(self, predictions):
        from keras import backend as K
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

    def predictions_and_gradient(self, image, label):
        predictions, gradient = self._pred_grad_fn([
            self._process_input(image[np.newaxis]),
            np.array([label])])
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(gradient)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == image.shape
        return predictions, gradient

    def backward(self, gradient, image):
        assert gradient.ndim == 1
        gradient = np.reshape(gradient, (-1, 1))
        gradient = self._bw_grad_fn([
            gradient,
            self._process_input(image[np.newaxis]),
        ])
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(gradient)
        assert gradient.shape == image.shape
        return gradient
