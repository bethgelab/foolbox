import eagerpy as ep

from .base import ModelWithPreprocessing


class JAXModel(ModelWithPreprocessing):
    def __init__(self, model, bounds, preprocessing=None):
        dummy = ep.jax.numpy.zeros(0)
        super().__init__(model, bounds, dummy, preprocessing=preprocessing)
