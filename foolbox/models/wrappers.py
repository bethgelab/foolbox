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
