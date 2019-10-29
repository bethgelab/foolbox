import tensorflow as tf


class TensorFlowModel:
    def __init__(self, model, bounds, preprocessing=None):
        assert tf.executing_eagerly()
        self._bounds = bounds
        self._model = model
        self._init_preprocessing(preprocessing)

    def _init_preprocessing(self, preprocessing):
        assert set(preprocessing.keys()) - {"mean", "std", "axis", "flip_axis"} == set()
        mean = self._preprocessing.get("mean", None)
        std = self._preprocessing.get("std", None)
        axis = self._preprocessing.get("axis", None)
        self._preprocessing_flip_axis = self._preprocessing.get("flip_axis", None)

        if mean is not None:
            mean = tf.convert_to_tensor(mean)
        if std is not None:
            std = tf.convert_to_tensor(std)

        if axis is not None:
            assert (
                axis < 0
            ), "axis must be negative integer, with -1 representing the last axis"
            shape = (1,) * (abs(axis) - 1)
            if mean is not None:
                assert (
                    mean.ndim == 1
                ), "If axis is specified, mean should be 1-dimensional"
                mean = tf.reshape(mean, mean.shape + shape)
            if std is not None:
                assert (
                    std.ndim == 1
                ), "If axis is specified, std should be 1-dimensional"
                std = tf.reshape(std, std.shape + shape)

        self._preprocessing_mean = mean
        self._preprocessing_std = std

    def _preprocess(self, inputs):
        x = inputs
        if self._preprocessing_flip_axis is not None:
            x = tf.reverse(x, self._preprocessing_flip_axis)
        if self._preprocessing_mean is not None:
            x = x - self._preprocessing_mean
        if self._preprocessing_std is not None:
            x = x / self._preprocessing_std
        assert x.dtype == inputs.dtype
        return x

    def bounds(self):
        return self._bounds

    def forward(self, inputs):
        x = inputs
        x = self._preprocess(x)
        x = self._model(x)
        assert x.ndim == 2
        return x

    def gradient(self, inputs, labels):
        x = x_ = inputs
        y = labels

        with tf.GradientTape() as tape:
            tape.watch(x)
            x = self._preprocess(x)
            x = self._model(x)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)

        grad = tape.gradient(loss, x_)
        assert grad.shape == x_.shape
        return grad
