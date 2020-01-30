import eagerpy as ep
from abc import ABC, abstractmethod
import copy
from typing import overload, Any

from ..devutils import wrap
from ..devutils import atleast_kd


class Model(ABC):
    @abstractmethod
    def bounds(self):
        ...

    @overload
    def forward(self, inputs: ep.Tensor) -> ep.Tensor:
        ...

    @overload  # noqa: F811
    def forward(self, inputs: Any) -> Any:
        ...

    @abstractmethod  # noqa: F811
    def forward(self, inputs):
        """Passes inputs through the model and returns the logits"""
        ...

    def transform_bounds(self, bounds) -> "Model":
        """Returns a new model with the desired bounds and updates the preprocessing accordingly"""
        return TransformBoundsWrapper(self, bounds)


class TransformBoundsWrapper(Model):
    def __init__(self, model, bounds):
        self._model = model
        self._bounds = bounds

    def bounds(self):
        return self._bounds

    def forward(self, inputs):
        inputs, restore = wrap(inputs)
        x = inputs
        x = self._preprocess(x)
        x = self._model.forward(x)
        return restore(x)

    def _preprocess(self, inputs):
        x = inputs

        # from bounds to (0, 1)
        min_, max_ = self._bounds
        x = (x - min_) / (max_ - min_)

        # from (0, 1) to wrapped model bounds
        min_, max_ = self._model.bounds()
        x = x * (max_ - min_) + min_
        return x


class ModelWithPreprocessing(Model):
    def __init__(self, model, bounds, dummy, preprocessing=None):
        if not callable(model):
            raise ValueError("expected model to be callable")  # pragma: no cover
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
        return restore(x)

    def transform_bounds(self, bounds, inplace=False):
        """Returns a new model with the desired bounds and updates the preprocessing accordingly"""
        # more efficient than the base class implementation because it avoids the additional wrapper
        a, b = self.bounds()
        c, d = bounds
        f = (d - c) / (b - a)

        mean, std, flip_axis = self._preprocess_args
        if mean is None:
            mean = 0.0
        mean = f * (mean - a) + c
        if std is None:
            std = 1.0
        std = f * std
        new_preprocess_args = (mean, std, flip_axis)

        if inplace:
            model = self
        else:
            model = copy.copy(self)
        model._bounds = bounds
        model._preprocess_args = new_preprocess_args
        return model

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
        unsupported = set(preprocessing.keys()) - {"mean", "std", "axis", "flip_axis"}
        if len(unsupported) > 0:
            raise ValueError(
                "found unsupported preprocessing keys: {}".format(
                    ", ".join(unsupported)
                )
            )
        mean = preprocessing.get("mean", None)
        std = preprocessing.get("std", None)
        axis = preprocessing.get("axis", None)
        flip_axis = preprocessing.get("flip_axis", None)

        if axis is not None and axis >= 0:
            raise ValueError("expected axis to be negative, -1 refers to the last axis")

        if mean is not None:
            try:
                mean = ep.astensor(mean)
            except ValueError:
                mean = ep.from_numpy(self.dummy, mean)
            if axis is not None:
                if mean.ndim != 1:
                    raise ValueError(
                        f"expected a 1D mean if axis is specified, got {mean.ndim}D"
                    )
                mean = atleast_kd(mean, -axis)
        if std is not None:
            try:
                std = ep.astensor(std)
            except ValueError:
                std = ep.from_numpy(self.dummy, std)
            if axis is not None:
                if std.ndim != 1:
                    raise ValueError(
                        f"expected a 1D std if axis is specified, got {std.ndim}D"
                    )
                std = atleast_kd(std, -axis)

        self._preprocess_args = (mean, std, flip_axis)
