from __future__ import absolute_import

import numpy as np
import warnings

from .base import DifferentiableModel


class PyTorchModel(DifferentiableModel):
    """Creates a :class:`Model` instance from a `PyTorch` module.

    Parameters
    ----------
    model : `torch.nn.Module`
        The PyTorch model that should be attacked.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    num_classes : int
        Number of classes for which the model will output predictions.
    channel_axis : int
        The index of the axis that represents color channels.
    device : string
        A string specifying the device to do computation on.
        If None, will default to "cuda:0" if torch.cuda.is_available()
        or "cpu" if not.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    """

    def __init__(
            self,
            model,
            bounds,
            num_classes,
            channel_axis=1,
            device=None,
            preprocessing=(0, 1)):

        # lazy import
        import torch

        super(PyTorchModel, self).__init__(bounds=bounds,
                                           channel_axis=channel_axis,
                                           preprocessing=preprocessing)

        self._num_classes = num_classes

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self._model = model.to(self.device)

        if model.training:
            warnings.warn(
                'The PyTorch model is in training mode and therefore might'
                ' not be deterministic. Call the eval() method to set it in'
                ' evaluation mode if this is not intended.')

    def forward(self, inputs):
        # lazy import
        import torch

        inputs, _ = self._process_input(inputs)
        n = len(inputs)
        inputs = torch.from_numpy(inputs).to(self.device)

        predictions = self._model(inputs)
        # TODO: add no_grad once we have a solution
        # for models that require grads internally
        # for inference
        # with torch.no_grad():
        #     predictions = self._model(inputs)
        predictions = predictions.detach().cpu().numpy()
        assert predictions.ndim == 2
        assert predictions.shape == (n, self.num_classes())
        return predictions

    def num_classes(self):
        return self._num_classes

    def forward_and_gradient_one(self, x, label):
        # lazy import
        import torch
        import torch.nn as nn

        input_shape = x.shape
        x, dpdx = self._process_input(x)
        target = np.array([label])
        target = torch.from_numpy(target).long().to(self.device)

        inputs = x[np.newaxis]
        inputs = torch.from_numpy(inputs).to(self.device)
        inputs.requires_grad_()

        predictions = self._model(inputs)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss.backward()
        grad = inputs.grad

        predictions = predictions.detach().cpu().numpy()
        predictions = np.squeeze(predictions, axis=0)
        assert predictions.ndim == 1
        assert predictions.shape == (self.num_classes(),)

        grad = grad.detach().cpu().numpy()
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return predictions, grad

    def gradient(self, inputs, labels):
        # lazy import
        import torch
        import torch.nn as nn

        input_shape = inputs.shape
        inputs, dpdx = self._process_input(inputs)
        target = np.asarray(labels)
        target = torch.from_numpy(labels).long().to(self.device)
        inputs = torch.from_numpy(inputs).to(self.device)
        inputs.requires_grad_()

        predictions = self._model(inputs)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss.backward()
        grad = inputs.grad
        grad = grad.detach().cpu().numpy()
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape
        return grad

    def _loss_fn(self, x, label):
        # lazy import
        import torch
        import torch.nn as nn

        x, _ = self._process_input(x)
        target = np.array([label])
        target = torch.from_numpy(target).long().to(self.device)
        inputs = torch.from_numpy(x[None]).to(self.device)
        predictions = self._model(inputs)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss = loss.cpu().numpy()
        return loss

    def backward(self, gradient, inputs):
        # lazy import
        import torch

        assert gradient.ndim == 2

        gradient = torch.from_numpy(gradient).to(self.device)

        input_shape = inputs.shape
        inputs, dpdx = self._process_input(inputs)
        inputs = torch.from_numpy(inputs).to(self.device)
        inputs.requires_grad_()
        predictions = self._model(inputs)

        assert gradient.dim() == 2
        assert predictions.dim() == 2
        assert gradient.size() == predictions.size()

        predictions.backward(gradient=gradient)

        grad = inputs.grad
        grad = grad.detach().cpu().numpy()
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape
        return grad
