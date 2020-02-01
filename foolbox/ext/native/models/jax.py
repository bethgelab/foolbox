import eagerpy as ep

from ..types import Bounds
from .base import ModelWithPreprocessing


class JAXModel(ModelWithPreprocessing):
    def __init__(self, model, bounds: Bounds, preprocessing: dict = None):
        dummy = ep.jax.numpy.zeros(0)
        super().__init__(model, bounds=bounds, dummy=dummy, preprocessing=preprocessing)
