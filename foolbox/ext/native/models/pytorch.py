import eagerpy as ep
import warnings

from .base import ModelWithPreprocessing


def get_device(device):
    import torch

    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


class PyTorchModel(ModelWithPreprocessing):
    def __init__(self, model, bounds, device=None, preprocessing=None):
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
        super().__init__(model, bounds, dummy, preprocessing=preprocessing)
        self.data_format = "channels_first"
