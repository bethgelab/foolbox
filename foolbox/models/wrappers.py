import eagerpy as ep

from ..types import Bounds

from .base import Model
from .base import T


class ThresholdingWrapper(Model):
    def __init__(self, model: Model, threshold: float):
        self._model = model
        self._threshold = threshold

    @property
    def bounds(self) -> Bounds:
        return self._model.bounds

    def __call__(self, inputs: T) -> T:
        min_, max_ = self._model.bounds
        x, restore_type = ep.astensor_(inputs)
        y = ep.where(x < self._threshold, min_, max_).astype(x.dtype)
        z = self._model(y)
        return restore_type(z)


class ExpectationOverTransformationWrapper(Model):
    def __init__(self, model: Model, n_steps: int = 16):
        self._model = model
        self._n_steps = n_steps

    @property
    def bounds(self) -> Bounds:
        return self._model.bounds

    def __call__(self, inputs: T) -> T:

        x, restore_type = ep.astensor_(inputs)

        for i in range(self._n_steps):
            z_t = self._model(x)

            if i == 0:
                z = z_t.expand_dims(0)
            else:
                z = ep.concatenate([z, z_t.expand_dims(0)], axis=0)

        z = z.mean(0)

        return restore_type(z)
