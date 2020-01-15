import jax.numpy as np
from jax.scipy.special import logsumexp
import jax
from .base import Model


def crossentropy(logits, labels):
    logprobs = logits - logsumexp(logits, axis=1, keepdims=True)
    nll = np.take_along_axis(logprobs, np.expand_dims(labels, axis=1), axis=1)
    ce = -np.mean(nll)
    return ce


class JAXModel(Model):
    def __init__(self, model, bounds, preprocessing=None):
        preprocess = self._create_preprocessing_fun(preprocessing)

        def f(x):
            x = preprocess(x)
            x = model(x)
            return x

        def loss(x, y):
            logits = f(x)
            return crossentropy(logits, y)

        g = jax.grad(loss)

        self._f = f
        self._g = g
        self._bounds = bounds

    def _create_preprocessing_fun(self, preprocessing):
        if preprocessing is None:
            preprocessing = dict()
        assert set(preprocessing.keys()) - {"mean", "std", "axis", "flip_axis"} == set()
        mean = preprocessing.get("mean", None)
        std = preprocessing.get("std", None)
        axis = preprocessing.get("axis", None)
        flip_axis = preprocessing.get("flip_axis", None)

        if mean is not None:
            mean = np.asarray(mean)
        if std is not None:
            std = np.asarray(std)

        if axis is not None:
            assert (
                axis < 0
            ), "axis must be negative integer, with -1 representing the last axis"
            shape = (1,) * (abs(axis) - 1)
            if mean is not None:
                assert (
                    mean.ndim == 1
                ), "If axis is specified, mean should be 1-dimensional"
                mean = mean.reshape(mean.shape + shape)
            if std is not None:
                assert (
                    std.ndim == 1
                ), "If axis is specified, std should be 1-dimensional"
                std = std.reshape(std.shape + shape)

        def preprocess(inputs):
            x = inputs
            if flip_axis is not None:
                x = np.flip(x, flip_axis)
            if mean is not None:
                x = x - mean
            if std is not None:
                x = x / std
            assert x.dtype == inputs.dtype
            return x

        return preprocess

    def bounds(self):
        return self._bounds

    def forward(self, inputs):
        logits = self._f(inputs)
        assert logits.ndim == 2
        return logits

    def gradient(self, inputs, labels):
        grad = self._g(inputs, labels)
        assert grad.shape == inputs.shape
        return grad
