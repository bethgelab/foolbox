from typing import TypeVar, Tuple, Any
import eagerpy as ep

from ..types import Bounds

from .base import Model


T = TypeVar("T")


class Foolbox2Model(Model):
    def __init__(self, model: Any) -> None:
        self._model = model

    @property
    def bounds(self) -> Bounds:
        bounds: Tuple[float, float] = self._model.bounds()
        return Bounds(*bounds)

    def __call__(self, inputs: T) -> T:
        x, restore_type = ep.astensor_(inputs)
        y = self._model.forward(x.numpy())
        z = ep.from_numpy(x, y)
        return restore_type(z)

    @property
    def data_format(self) -> str:
        channel_axis = self._model.channel_axis()
        if channel_axis == 1:
            data_format = "channels_first"
        elif channel_axis == 3:
            data_format = "channels_last"
        return data_format
