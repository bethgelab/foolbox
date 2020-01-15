import foolbox
from .base import Model


class FoolboxModel(Model):
    def __init__(self, model):
        assert isinstance(model, foolbox.models.base.Model)
        self._model = model

    def bounds(self):
        return self._model.bounds()

    def forward(self, inputs):
        return self._model.forward(inputs)

    def gradient(self, inputs, labels):
        return self._model.gradient(inputs, labels)

    @property
    def foolbox_model(self):
        return self._model
