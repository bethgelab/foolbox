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

    def batch_predictions(self, images):
        # lazy import
        import torch

        images, _ = self._process_input(images)
        n = len(images)
        images = torch.from_numpy(images).to(self.device)

        predictions = self._model(images)
        # TODO: add no_grad once we have a solution
        # for models that require grads internally
        # for inference
        # with torch.no_grad():
        #     predictions = self._model(images)
        predictions = predictions.detach().cpu().numpy()
        assert predictions.ndim == 2
        assert predictions.shape == (n, self.num_classes())
        return predictions

    def num_classes(self):
        return self._num_classes

    def predictions_and_gradient(self, image, label):
        # lazy import
        import torch
        import torch.nn as nn

        input_shape = image.shape
        image, dpdx = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target).long().to(self.device)

        images = image[np.newaxis]
        images = torch.from_numpy(images).to(self.device)
        images.requires_grad_()

        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss.backward()
        grad = images.grad

        predictions = predictions.detach().cpu().numpy()
        predictions = np.squeeze(predictions, axis=0)
        assert predictions.ndim == 1
        assert predictions.shape == (self.num_classes(),)

        grad = grad.detach().cpu().numpy()
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return predictions, grad

    def batch_gradients(self, images, labels):
        # lazy import
        import torch
        import torch.nn as nn

        input_shape = images.shape
        images, dpdx = self._process_input(images)
        target = np.asarray(labels)
        target = torch.from_numpy(labels).long().to(self.device)
        images = torch.from_numpy(images).to(self.device)
        images.requires_grad_()

        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss.backward()
        grad = images.grad
        grad = grad.detach().cpu().numpy()
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape
        return grad

    def _loss_fn(self, image, label):
        # lazy import
        import torch
        import torch.nn as nn

        image, _ = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target).long().to(self.device)
        images = torch.from_numpy(image[None]).to(self.device)
        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss = loss.cpu().numpy()
        return loss

    def batch_backward(self, gradients, images):
        # lazy import
        import torch

        assert gradients.ndim == 2

        gradients = torch.from_numpy(gradients).to(self.device)

        input_shape = images.shape
        images, dpdx = self._process_input(images)
        images = torch.from_numpy(images).to(self.device)
        images.requires_grad_()
        predictions = self._model(images)

        assert gradients.dim() == 2
        assert predictions.dim() == 2
        assert gradients.size() == predictions.size()

        predictions.backward(gradient=gradients)

        grad = images.grad
        grad = grad.detach().cpu().numpy()
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape
        return grad
