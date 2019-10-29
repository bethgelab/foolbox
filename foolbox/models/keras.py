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
        Tuple of lower and upper bound for the pixel values, usually (0, 1) or (0, 255).
    channel_axis : int or 'auto'
        The index of the axis that represents color channels. If 'auto',
        will be set automatically based on keras.backend.image_data_format()
    preprocessing: dict or tuple
        Can be a tuple with two elements representing mean and standard
        deviation or a dict with keys "mean" and "std". The two elements
        should be floats or numpy arrays. "mean" is subtracted from the input,
        the result is then divided by "std". If "mean" and "std" are
        1-dimensional arrays, an additional (negative) "axis" key can be
        given such that "mean" and "std" will be broadcasted to that axis
        (typically -1 for "channels_last" and -3 for "channels_first", but
        might be different when using e.g. 1D convolutions). Finally,
        a (negative) "flip_axis" can be specified. This axis will be flipped
        (before "mean" is subtracted), e.g. to convert RGB to BGR.
    predicts : str
        Specifies whether the `Keras` model predicts logits or probabilities.
        Logits are preferred, but probabilities are the default.

    """

    def __init__(
        self,
        model,
        bounds,
        channel_axis="auto",
        preprocessing=(0, 1),
        predicts="probabilities",
    ):
        from keras import backend as K

        if channel_axis == "auto":
            if K.image_data_format() == "channels_first":
                channel_axis = 1
            else:
                channel_axis = 3

        super(KerasModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocessing=preprocessing
        )

        import keras
        from pkg_resources import parse_version

        assert parse_version(keras.__version__) >= parse_version(
            "2.0.7"
        ), "Keras version needs to be 2.0.7 or newer"

        if predicts == "probs":
            predicts = "probabilities"
        assert predicts in ["probabilities", "logits"]

        inputs = model.input
        labels = K.placeholder(shape=(None,))
        predictions = model.output

        shape = K.int_shape(predictions)
        _, num_classes = shape
        assert num_classes is not None

        self._num_classes = num_classes

        if predicts == "probabilities":
            if K.backend() == "tensorflow":
                (predictions,) = predictions.op.inputs
                loss = K.sparse_categorical_crossentropy(
                    labels, predictions, from_logits=True
                )
            else:  # pragma: no cover
                logging.warning(
                    "relying on numerically unstable conversion from probabilities to softmax"
                )
                loss = K.sparse_categorical_crossentropy(
                    labels, predictions, from_logits=False
                )

                # transform the probability predictions into logits, so that
                # the rest of this code can assume predictions to be logits
                predictions = self._to_logits(predictions)
        elif predicts == "logits":
            loss = K.sparse_categorical_crossentropy(
                labels, predictions, from_logits=True
            )

        loss = K.sum(loss, axis=0)
        (gradient,) = K.gradients(loss, [inputs])

        backward_grad_logits = K.placeholder(shape=predictions.shape)
        backward_loss = K.sum(
            K.batch_dot(predictions, backward_grad_logits, axes=-1), axis=0
        )
        (backward_grad_inputs,) = K.gradients(backward_loss, [inputs])

        self._loss_fn = K.function([inputs, labels], [loss])
        self._forward_fn = K.function([inputs], [predictions])
        self._gradient_fn = K.function([inputs, labels], [gradient])
        self._backward_fn = K.function(
            [backward_grad_logits, inputs], [backward_grad_inputs]
        )
        self._forward_and_gradient_fn = K.function(
            [inputs, labels], [predictions, gradient]
        )

    def _to_logits(self, predictions):  # pragma: no cover
        from keras import backend as K

        eps = 10e-8
        predictions = K.clip(predictions, eps, 1 - eps)
        predictions = K.log(predictions)
        return predictions

    def num_classes(self):
        return self._num_classes

    def forward(self, inputs):
        px, _ = self._process_input(inputs)
        (predictions,) = self._forward_fn([px])
        assert predictions.shape == (inputs.shape[0], self.num_classes())
        return predictions

    def forward_and_gradient_one(self, x, label):
        input_shape = x.shape
        px, dpdx = self._process_input(x)
        predictions, gradient = self._forward_and_gradient_fn(
            [px[np.newaxis], np.asarray(label)[np.newaxis]]
        )
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == input_shape
        return predictions, gradient

    def forward_and_gradient(self, inputs, labels):
        inputs_shape = inputs.shape
        inputs, dpdx = self._process_input(inputs)
        labels = np.asarray(labels)
        predictions, gradient = self._forward_and_gradient_fn([inputs, labels])
        gradient = self._process_gradient(dpdx, gradient)
        assert predictions.shape == (len(inputs), self.num_classes())
        assert gradient.shape == inputs_shape
        return predictions, gradient

    def gradient(self, inputs, labels):
        px, dpdx = self._process_input(inputs)
        (g,) = self._gradient_fn([px, labels])
        g = self._process_gradient(dpdx, g)
        assert g.shape == inputs.shape
        return g

    def backward(self, gradient, inputs):
        assert gradient.ndim == 2
        px, dpdx = self._process_input(inputs)
        (g,) = self._backward_fn([gradient, px])
        g = self._process_gradient(dpdx, g)
        assert g.shape == inputs.shape
        return g
