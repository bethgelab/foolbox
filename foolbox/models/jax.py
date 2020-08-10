from typing import Any, Optional
import eagerpy as ep

from ..types import BoundsInput, Preprocessing

from .base import ModelWithPreprocessing


class JAXModel(ModelWithPreprocessing):
    def __init__(
        self,
        model: Any,
        bounds: BoundsInput,
        preprocessing: Preprocessing = None,
        data_format: Optional[str] = "channels_last",
    ):
        dummy = ep.jax.numpy.zeros(0)
        super().__init__(model, bounds=bounds, dummy=dummy, preprocessing=preprocessing)
        self._data_format = data_format

    @property
    def data_format(self) -> str:
        if self._data_format is None:
            raise AttributeError(  # AttributeError -> hasattr returns False
                "please specify data_format when initializing the JaxModel"
            )
        return self._data_format
