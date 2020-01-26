import eagerpy as ep
from abc import ABC, abstractmethod

from ..devutils import wrap
from ..devutils import atleast_kd


class Model(ABC):
    @abstractmethod
    def bounds(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs):
        """Passes inputs through the model and returns the logits"""
        raise NotImplementedError


class ModelWithPreprocessing(Model):
    def __init__(self, model, bounds, dummy, preprocessing=None):
        assert callable(model)
        self._model = model
        self._bounds = bounds
        self.dummy = dummy
        self._init_preprocessing(preprocessing)

    def bounds(self):
        return self._bounds

    def forward(self, inputs):
        inputs, restore = wrap(inputs)
        x = inputs
        x = self._preprocess(x)
        x = ep.astensor(self._model(x.tensor))
        assert x.ndim == 2
        return restore(x)

    def _preprocess(self, inputs):
        x = inputs
        mean, std, flip_axis = self._preprocess_args
        if flip_axis is not None:
            x = x.flip(axis=flip_axis)
        if mean is not None:
            x = x - mean
        if std is not None:
            x = x / std
        assert x.dtype == inputs.dtype
        return x

    def _init_preprocessing(self, preprocessing):
        if preprocessing is None:
            preprocessing = dict()
        assert set(preprocessing.keys()) - {"mean", "std", "axis", "flip_axis"} == set()
        mean = preprocessing.get("mean", None)
        std = preprocessing.get("std", None)
        axis = preprocessing.get("axis", None)
        flip_axis = preprocessing.get("flip_axis", None)

        if axis is not None:
            assert axis < 0, "expected axis to be negative, -1 refers to the last axis"

        if mean is not None:
            mean = ep.from_numpy(self.dummy, mean)
            if axis is not None:
                assert (
                    mean.ndim == 1
                ), f"expected a 1D mean if axis is specified, got {mean.ndim}D"
                mean = atleast_kd(mean, -axis)
        if std is not None:
            std = ep.from_numpy(self.dummy, std)
            if axis is not None:
                assert (
                    std.ndim == 1
                ), f"expected a 1D std if axis is specified, got {std.ndim}D"
                std = atleast_kd(std, -axis)

        self._preprocess_args = (mean, std, flip_axis)
