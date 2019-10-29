import tensorflow as tf


class TensorFlowModel:
    def __init__(self, model, bounds, preprocessing=dict(mean=0, std=1)):
        assert tf.executing_eagerly()
        assert set(preprocessing.keys()) - {"mean", "std"} == set()
        self._bounds = bounds
        self._preprocessing = preprocessing
        self._model = model

    def _preprocess(self, inputs):
        x = inputs

        mean = self._preprocessing.get("mean", 0)
        if mean != 0:
            x = x - mean

        std = self._preprocessing.get("std", 1)
        if std != 1:
            x = x / std

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
