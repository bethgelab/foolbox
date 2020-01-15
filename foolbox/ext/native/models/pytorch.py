import torch
import torch.nn.functional as F
import warnings
from .base import Model


class PyTorchModel(Model):
    def __init__(self, model, bounds, device=None, preprocessing=None):
        self._bounds = bounds

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if model.training:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The PyTorch model is in training mode and therefore might"
                    " not be deterministic. Call the eval() method to set it in"
                    " evaluation mode if this is not intended."
                )
        self._model = model.to(self.device)
        self._init_preprocessing(preprocessing)

    def _init_preprocessing(self, preprocessing):
        if preprocessing is None:
            preprocessing = dict()
        assert set(preprocessing.keys()) - {"mean", "std", "axis", "flip_axis"} == set()
        mean = preprocessing.get("mean", None)
        std = preprocessing.get("std", None)
        axis = preprocessing.get("axis", None)
        flip_axis = preprocessing.get("flip_axis", None)

        if mean is not None:
            mean = torch.as_tensor(mean).to(self.device)
        if std is not None:
            std = torch.as_tensor(std).to(self.device)

        if axis is not None:
            assert (
                axis < 0
            ), "axis must be negative integer, with -1 representing the last axis"
            shape = (1,) * (abs(axis) - 1)
            if mean is not None:
                assert (
                    mean.ndim == 1
                ), "If axis is specified, mean should be 1-dimensional"
                mean = mean.reshape(mean.shape + shape)
            if std is not None:
                assert (
                    std.ndim == 1
                ), "If axis is specified, std should be 1-dimensional"
                std = std.reshape(std.shape + shape)

        self._preprocessing_mean = mean
        self._preprocessing_std = std
        self._preprocessing_flip_axis = flip_axis

    def _preprocess(self, inputs):
        x = inputs
        if self._preprocessing_flip_axis is not None:
            x = torch.flip(x, (self._preprocessing_flip_axis,))
        if self._preprocessing_mean is not None:
            x = x - self._preprocessing_mean
        if self._preprocessing_std is not None:
            x = x / self._preprocessing_std
        assert x.dtype == inputs.dtype
        return x

    def bounds(self):
        return self._bounds

    def forward(self, inputs):
        x = inputs
        assert x.device == self.device
        x = self._preprocess(x)
        x = self._model(x)
        assert x.ndim == 2
        return x

    def gradient(self, inputs, labels):
        x = inputs.clone()
        x.requires_grad_()
        x_ = x
        y = labels
        assert x.device == self.device
        assert y.device == self.device
        x = self.forward(x)
        loss = F.cross_entropy(x, y)
        loss.backward()
        grad = x_.grad
        assert grad.shape == x_.shape
        return grad
