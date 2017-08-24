import numpy as np

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

    def batch_predictions(self, images):
        # lazy import
        import torch
        from torch.autograd import Variable

        images = self._process_input(images)
        images = self._torch(images, volatile=True)
        predictions = self._model(images)
        predictions = self._numpy(predictions)
        assert predictions.ndim == 2
        assert predictions.shape == (len(images), self.num_classes())
        return predictions

    def num_classes(self):
        return self._num_classes

    def _loss(self, loss, predictions, label):
        # lazy import
        import torch
        from torch.autograd import Variable

        if hasattr(loss, '__call__'):
            return loss(predictions, label)
        elif loss == 'crossentropy':
            target = self._torch(np.array([label]))
            ce = torch.nn.CrossEntropyLoss()
            return ce(predictions, target)
        elif loss is None or loss == 'logits':
            return -predictions[0, label]
        elif loss == 'carlini':
            loss = torch.max(predictions[0, :]) - predictions[0, label]
            return torch.nn.functional.relu(loss)
        else:
            raise NotImplementedError('The loss {} is currently not \
                    implemented for this framework.'.format(loss))

    def predictions_and_gradient(self, image, label, loss=None):
        # lazy import
        import torch
        from torch.autograd import Variable

        assert image.ndim == 3
        image  = self._process_input(image)
        images = self._torch(image[np.newaxis], requires_grad=True)
        predictions = self._model(images)
        loss = self._loss(loss, predictions, label)
        loss.backward()
        grad = images.grad

        predictions = self._numpy(predictions)
        predictions = np.squeeze(predictions, axis=0)
        assert predictions.ndim == 1
        assert predictions.shape == (self.num_classes(),)

        grad = self._numpy(grad)
        grad = self._process_gradient(grad)
        grad = np.squeeze(grad, axis=0)
        assert grad.shape == image.shape

        return predictions, grad

    def _loss_fn(self, image, label, loss=None):
        image  = self._process_input(image)
        images = self._torch(image[None], volatile=True)
        predictions = self._model(images)

        loss = self._loss(loss, predictions, label)
        return self._numpy(loss)

    def _torch(self, npvar, volatile=False, requires_grad=False):
        """ Turns numpy array into Torch variable """
        # lazy import
        import torch
        from torch.autograd import Variable

        var = torch.from_numpy(npvar)
        if self.cuda:  # pragma: no cover
            var = var.cuda()

        return Variable(var, volatile=volatile, requires_grad=requires_grad)

    def _numpy(self, var):
        """ Returns numpy value of Torch tensor """
        var = var.data
        if self.cuda:  # pragma: no cover
            var = var.cpu()
        return var.numpy()
