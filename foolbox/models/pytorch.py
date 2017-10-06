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
    cuda : bool
        A boolean specifying whether the model uses CUDA.
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
            cuda=True,
            preprocessing=(0, 1)):

        super(PyTorchModel, self).__init__(bounds=bounds,
                                           channel_axis=channel_axis,
                                           preprocessing=preprocessing)

        self._num_classes = num_classes
        self._model = model
        self.cuda = cuda

        if model.training:
            warnings.warn(
                'The PyTorch model is in training mode and therefore might'
                ' not be deterministic. Call the eval() method to set it in'
                ' evaluation mode if this is not intended.')

    def batch_predictions(self, images):
        # lazy import
        import torch
        from torch.autograd import Variable

        images = self._process_input(images)
        n = len(images)
        images = torch.from_numpy(images)
        if self.cuda:  # pragma: no cover
            images = images.cuda()
        images = Variable(images, volatile=True)
        predictions = self._model(images)
        predictions = predictions.data
        if self.cuda:  # pragma: no cover
            predictions = predictions.cpu()
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
        from torch.autograd import Variable

        image = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target)
        if self.cuda:  # pragma: no cover
            target = target.cuda()
        target = Variable(target)

        assert image.ndim == 3
        images = image[np.newaxis]
        images = torch.from_numpy(images)
        if self.cuda:  # pragma: no cover
            images = images.cuda()
        images = Variable(images, requires_grad=True)
        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss.backward()
        grad = images.grad

        predictions = predictions.data
        if self.cuda:  # pragma: no cover
            predictions = predictions.cpu()

        predictions = predictions.numpy()
        predictions = np.squeeze(predictions, axis=0)
        assert predictions.ndim == 1
        assert predictions.shape == (self.num_classes(),)

        grad = grad.data
        if self.cuda:  # pragma: no cover
            grad = grad.cpu()
        grad = grad.numpy()
        grad = self._process_gradient(grad)
        grad = np.squeeze(grad, axis=0)
        assert grad.shape == image.shape

        return predictions, grad

    def _loss_fn(self, image, label):
        # lazy import
        import torch
        import torch.nn as nn
        from torch.autograd import Variable

        image = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target)
        if self.cuda:  # pragma: no cover
            target = target.cuda()
        target = Variable(target)

        images = torch.from_numpy(image[None])
        if self.cuda:  # pragma: no cover
            images = images.cuda()
        images = Variable(images, volatile=True)
        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss = loss.data
        if self.cuda:  # pragma: no cover
            loss = loss.cpu()
        loss = loss.numpy()
        return loss

    def backward(self, gradient, image):
        # lazy import
        import torch
        from torch.autograd import Variable

        assert gradient.ndim == 1

        gradient = torch.from_numpy(gradient)
        if self.cuda:  # pragma: no cover
            gradient = gradient.cuda()
        gradient = Variable(gradient)

        image = self._process_input(image)
        assert image.ndim == 3
        images = image[np.newaxis]
        images = torch.from_numpy(images)
        if self.cuda:  # pragma: no cover
            images = images.cuda()
        images = Variable(images, requires_grad=True)
        predictions = self._model(images)

        print(predictions.size())
        predictions = predictions[0]

        assert gradient.dim() == 1
        assert predictions.dim() == 1
        assert gradient.size() == predictions.size()

        loss = torch.dot(predictions, gradient)
        loss.backward()
        # should be the same as predictions.backward(gradient=gradient)

        grad = images.grad

        grad = grad.data
        if self.cuda:  # pragma: no cover
            grad = grad.cpu()
        grad = grad.numpy()
        grad = self._process_gradient(grad)
        grad = np.squeeze(grad, axis=0)
        assert grad.shape == image.shape

        return grad
