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

    def _old_pytorch(self):
        # lazy import
        import torch
        version = torch.__version__.split('.')[:2]
        pre04 = int(version[0]) == 0 and int(version[1]) < 4
        return pre04

    def batch_predictions(self, images):
        # lazy import
        import torch
        if self._old_pytorch():  # pragma: no cover
            from torch.autograd import Variable

        images, _ = self._process_input(images)
        n = len(images)
        images = torch.from_numpy(images).to(self.device)

        if self._old_pytorch():  # pragma: no cover
            images = Variable(images, volatile=True)
            predictions = self._model(images)
            predictions = predictions.data
        else:
            predictions = self._model(images)
            # TODO: add no_grad once we have a solution
            # for models that require grads internally
            # for inference
            # with torch.no_grad():
            #     predictions = self._model(images)
        predictions = predictions.to("cpu")
        if not self._old_pytorch():
            predictions = predictions.detach()
        predictions = predictions.numpy()
        assert predictions.ndim == 2
        assert predictions.shape == (n, self.num_classes())
        return predictions

    def num_classes(self):
        return self._num_classes

    def predictions_and_gradient(self, image, label):
        # lazy import
        import torch
        import torch.nn as nn
        if self._old_pytorch():  # pragma: no cover
            from torch.autograd import Variable

        input_shape = image.shape
        image, dpdx = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target).to(self.device)

        images = image[np.newaxis]
        images = torch.from_numpy(images).to(self.device)

        if self._old_pytorch():  # pragma: no cover
            target = Variable(target)
            images = Variable(images, requires_grad=True)
        else:
            images.requires_grad_()

        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss.backward()
        grad = images.grad

        if self._old_pytorch():  # pragma: no cover
            predictions = predictions.data
        predictions = predictions.to("cpu")

        if not self._old_pytorch():
            predictions = predictions.detach()
        predictions = predictions.numpy()
        predictions = np.squeeze(predictions, axis=0)
        assert predictions.ndim == 1
        assert predictions.shape == (self.num_classes(),)

        if self._old_pytorch():  # pragma: no cover
            grad = grad.data
        grad = grad.to("cpu")
        if not self._old_pytorch():
            grad = grad.detach()
        grad = grad.numpy()
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return predictions, grad

    def _loss_fn(self, image, label):
        # lazy import
        import torch
        import torch.nn as nn
        if self._old_pytorch():  # pragma: no cover
            from torch.autograd import Variable

        image, _ = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target).to(self.device)
        if self._old_pytorch():  # pragma: no cover
            target = Variable(target)

        images = torch.from_numpy(image[None]).to(self.device)
        if self._old_pytorch():  # pragma: no cover
            images = Variable(images, volatile=True)
        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        if self._old_pytorch():  # pragma: no cover
            loss = loss.data
        loss = loss.to("cpu")
        loss = loss.numpy()
        return loss

    def backward(self, gradient, image):
        # lazy import
        import torch
        if self._old_pytorch():  # pragma: no cover
            from torch.autograd import Variable

        assert gradient.ndim == 1

        gradient = torch.from_numpy(gradient).to(self.device)
        if self._old_pytorch():  # pragma: no cover
            gradient = Variable(gradient)

        input_shape = image.shape
        image, dpdx = self._process_input(image)
        images = image[np.newaxis]
        images = torch.from_numpy(images).to(self.device)
        if self._old_pytorch():  # pragma: no cover
            images = Variable(images, requires_grad=True)
        else:
            images.requires_grad_()
        predictions = self._model(images)

        predictions = predictions[0]

        assert gradient.dim() == 1
        assert predictions.dim() == 1
        assert gradient.size() == predictions.size()

        loss = torch.dot(predictions, gradient)
        loss.backward()
        # should be the same as predictions.backward(gradient=gradient)

        grad = images.grad

        if self._old_pytorch():  # pragma: no cover
            grad = grad.data
        grad = grad.to("cpu")
        if not self._old_pytorch():
            grad = grad.detach()
        grad = grad.numpy()
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return grad
