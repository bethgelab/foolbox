import jax.numpy as np
from .base import Model


class JAXModel(Model):
    def __init__(self, model, bounds, preprocessing=None):
        self._bounds = bounds
        self._model = model
        self._preprocess = self._create_preprocessing_fun(preprocessing)

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
        x = inputs
        x = self._preprocess(x)
        x = self._model(x)
        assert x.ndim == 2
        return x
