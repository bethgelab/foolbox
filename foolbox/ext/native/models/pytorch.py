import torch
import torch.nn.functional as F
import warnings


class PyTorchModel:
    def __init__(self, model, bounds, device=None, preprocessing=dict(mean=0, std=1)):
        assert set(preprocessing.keys()) - {"mean", "std", "axis"} == set()
        self._bounds = bounds
        self._preprocessing = preprocessing

        if model.training:
            warnings.warn(
                "The PyTorch model is in training mode and therefore might"
                " not be deterministic. Call the eval() method to set it in"
                " evaluation mode if this is not intended."
            )

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self._model = model.to(self.device)

    def _preprocess(self, inputs):
        x = inputs

        mean = self._preprocessing.get("mean", 0)
        std = self._preprocessing.get("std", 1)
        axis = self._preprocessing.get("axis", None)

        if axis is not None:
            mean = torch.as_tensor(mean)
            std = torch.as_tensor(std)
            assert mean.ndim == 1, "If axis is specified, mean should be 1-dimensional"
            assert std.ndim == 1, "If axis is specified, std should be 1-dimensional"
            assert (
                axis < 0
            ), "axis must be negative integer, with -1 representing the last axis"
            s = (1,) * (abs(axis) - 1)
            mean = mean.reshape(mean.shape + s)
            std = std.reshape(std.shape + s)

        if isinstance(mean, torch.Tensor) or mean != 0:
            x = x - mean

        if isinstance(std, torch.Tensor) or std != 1:
            x = x / std

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
        x = self._preprocess(x)
        x = self._model(x)
        loss = F.cross_entropy(x, y)
        loss.backward()
        grad = x_.grad
        assert grad.shape == x_.shape
        return grad
