from .base import Model
from ..devutils import unwrap


class Foolbox2Model(Model):
    def __init__(self, model):
        self._model = model

    @property
    def bounds(self):
        return self._model.bounds()

    def __call__(self, inputs):
        inputs, restore = unwrap(inputs)
        return restore(self._model.forward(inputs))

    @property
    def data_format(self):
        channel_axis = self._model.channel_axis()
        if channel_axis == 1:
            data_format = "channels_first"
        elif channel_axis == 3:
            data_format = "channels_last"
        return data_format
