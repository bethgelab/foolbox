import numpy as np
import warnings

from .base import DifferentiableModel


class PyTorchModel(DifferentiableModel):
    """Creates a :class:`Model` instance from a `PyTorch` module.

    Parameters
    ----------
    model : `torch.nn.Module`
        The PyTorch model that should be attacked. It should predict logits
        or log-probabilities, i.e. predictions without the softmax.
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
    preprocessing: dict or tuple
        Can be a tuple with two elements representing mean and standard
        deviation or a dict with keys "mean" and "std". The two elements
        should be floats or numpy arrays. "mean" is subtracted from the input,
        the result is then divided by "std". If "mean" and "std" are
        1-dimensional arrays, an additional (negative) "axis" key can be
        given such that "mean" and "std" will be broadcasted to that axis
        (typically -1 for "channels_last" and -3 for "channels_first", but
        might be different when using e.g. 1D convolutions). Finally,
        a (negative) "flip_axis" can be specified. This axis will be flipped
        (before "mean" is subtracted), e.g. to convert RGB to BGR.

    """

    def __init__(
        self,
        model,
        bounds,
        num_classes,
        channel_axis=1,
        device=None,
        preprocessing=(0, 1),
    ):

        # lazy import
        import torch

        super(PyTorchModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocessing=preprocessing
        )

        self._num_classes = num_classes

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self._model = model.to(self.device)

        if model.training:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The PyTorch model is in training mode and therefore might"
                    " not be deterministic. Call the eval() method to set it in"
                    " evaluation mode if this is not intended."
                )

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

    def forward_and_gradient(self, inputs, labels):
        # lazy import
        import torch
        import torch.nn as nn

        inputs_shape = inputs.shape
        inputs, dpdx = self._process_input(inputs)
        labels = np.array(labels)
        labels = torch.from_numpy(labels).long().to(self.device)

        inputs = torch.from_numpy(inputs).to(self.device)
        inputs.requires_grad_()

        predictions = self._model(inputs)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, labels)
        loss.backward()
        grad = inputs.grad

        predictions = predictions.detach().cpu().numpy()
        assert predictions.ndim == 2
        assert predictions.shape == (len(inputs), self.num_classes())

        grad = grad.detach().cpu().numpy()
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == inputs_shape

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

        # if x and label were already batched, make sure that we remove
        # the added dimension again
        if len(target.shape) == 2:
            target = target[0]
            inputs = inputs[0]

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
