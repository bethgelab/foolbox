from __future__ import absolute_import

from .base import DifferentiableModel

import numpy as np


class MXNetGluonModel(DifferentiableModel):
    """Creates a :class:`Model` instance from existing `MXNet` symbols and weights.

    Parameters
    ----------
    data : `mxnet.symbol.Variable`
        The input to the model.
    logits : `mxnet.symbol.Symbol`
        The predictions of the model, before the softmax.
    args : `dictionary mapping str to mxnet.nd.array`
        The parameters of the model.
    ctx : `mxnet.context.Context`
        The device, e.g. mxnet.cpu() or mxnet.gpu().
    num_classes : int
        The number of classes.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    aux_states : `dictionary mapping str to mxnet.nd.array`
        The states of auxiliary parameters of the model.
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

    def batch_predictions(self, image):
        import mxnet as mx
        image = self._process_input(image)
        data_array = mx.nd.array(image, ctx=self._device)
        data_array.attach_grad()
        with mx.autograd.record(train_mode=False):
            L = self._block(data_array)
        return np.squeeze(L.asnumpy(), axis=0)

    def predictions(self, image):
        return self.batch_predictions(image)

    def predictions_and_gradient(self, image, label):
        import mxnet as mx
        image = self._process_input(image)
        label = mx.nd.array([label])
        data_array = mx.nd.array(image, ctx=self._device)
        data_array.attach_grad()
        with mx.autograd.record(train_mode=False):
            L = self._block(data_array)
            loss = mx.nd.softmax_cross_entropy(L, label)
            loss.backward()
        return np.squeeze(L.asnumpy(), axis=0),
        self._process_gradient(data_array.grad.asnumpy())

    def _loss_fn(self, image, label):
        import mxnet as mx
        image = self._process_input(image)
        label = mx.nd.array([label])
        data_array = mx.nd.array(image, ctx=self._device)
        data_array.attach_grad()
        with mx.autograd.record(train_mode=False):
            L = self._block(data_array)
            loss = mx.nd.softmax_cross_entropy(L, label)
            loss.backward()
        return loss.asnumpy()
