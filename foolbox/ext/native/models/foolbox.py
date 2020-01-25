from .base import Model
from ..devutils import unwrap


class Foolbox2Model(Model):
    def __init__(self, model):
        self._model = model

    def bounds(self):
        return self._model.bounds()

    def forward(self, inputs):
        inputs, restore = unwrap(inputs)
        return restore(self._model.forward(inputs))
