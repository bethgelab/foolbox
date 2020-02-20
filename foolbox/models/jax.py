from typing import Any
import eagerpy as ep

from ..types import BoundsInput, Preprocessing

from .base import ModelWithPreprocessing


class JAXModel(ModelWithPreprocessing):
    def __init__(
        self, model: Any, bounds: BoundsInput, preprocessing: Preprocessing = None
    ):
        dummy = ep.jax.numpy.zeros(0)
        super().__init__(model, bounds=bounds, dummy=dummy, preprocessing=preprocessing)
