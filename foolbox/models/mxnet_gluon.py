from __future__ import absolute_import

from .base import DifferentiableModel

import numpy as np


class MXNetGluonModel(DifferentiableModel):
    """Creates a :class:`Model` instance from an existing `MXNet Gluon` Block.

    Parameters
    ----------
    block : `mxnet.gluon.Block`
        The Gluon Block representing the model to be run.
    ctx : `mxnet.context.Context`
        The device, e.g. mxnet.cpu() or mxnet.gpu().
    num_classes : int
        The number of classes.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.

    """

    def __init__(
            self,
            block,
            bounds,
            num_classes,
            ctx=None,
            channel_axis=1,
            preprocessing=(0, 1)):
        import mxnet as mx
        self._num_classes = num_classes

        if ctx is None:
            ctx = mx.cpu()

        super(MXNetGluonModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self._device = ctx
        self._block = block

    def num_classes(self):
        return self._num_classes

    def batch_predictions(self, images):
        import mxnet as mx
        images, _ = self._process_input(images)
        data_array = mx.nd.array(images, ctx=self._device)
        data_array.attach_grad()
        with mx.autograd.record(train_mode=False):
            L = self._block(data_array)
        return L.asnumpy()

    def predictions_and_gradient(self, image, label):
        import mxnet as mx
        image, dpdx = self._process_input(image)
        label = mx.nd.array([label])
        data_array = mx.nd.array(image[np.newaxis], ctx=self._device)
        data_array.attach_grad()
        with mx.autograd.record(train_mode=False):
            logits = self._block(data_array)
            loss = mx.nd.softmax_cross_entropy(logits, label)
            loss.backward()
        predictions = np.squeeze(logits.asnumpy(), axis=0)
        gradient = np.squeeze(data_array.grad.asnumpy(), axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        return predictions, gradient

    def _loss_fn(self, image, label):
        import mxnet as mx
        image, _ = self._process_input(image)
        label = mx.nd.array([label])
        data_array = mx.nd.array(image[np.newaxis], ctx=self._device)
        data_array.attach_grad()
        with mx.autograd.record(train_mode=False):
            logits = self._block(data_array)
            loss = mx.nd.softmax_cross_entropy(logits, label)
            loss.backward()
        return loss.asnumpy()

    def backward(self, gradient, image):  # pragma: no cover
        # TODO: backward functionality has not yet been implemented
        # for MXNetGluonModel
        raise NotImplementedError
