from typing import Any
import warnings
import eagerpy as ep

from ..types import BoundsInput
from .base import ModelWithPreprocessing


def get_device(device) -> Any:
    import torch

    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


class PyTorchModel(ModelWithPreprocessing):
    def __init__(
        self, model, bounds: BoundsInput, device=None, preprocessing: dict = None
    ) -> None:
        import torch

        if model.training:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The PyTorch model is in training mode and therefore might"
                    " not be deterministic. Call the eval() method to set it in"
                    " evaluation mode if this is not intended."
                )

        device = get_device(device)
        model = model.to(device)
        dummy = ep.torch.zeros(0, device=device)

        # we need to make sure the output only requires_grad if the input does
        def _model(x):
            with torch.set_grad_enabled(x.requires_grad):
                return model(x)

        super().__init__(
            _model, bounds=bounds, dummy=dummy, preprocessing=preprocessing
        )

        self.data_format = "channels_first"
        self.device = device
