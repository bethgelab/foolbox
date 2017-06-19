from __future__ import absolute_import

import numpy as np

from .base import DifferentiableModel


class MXNetModel(DifferentiableModel):
    """Creates a :class:`Model` instance from existing `MXNet` symbols and weights.

    Parameters
    ----------
    data : `mxnet.symbol.Variable`
        The input to the model.
    logits : `mxnet.symbol.Symbol`
        The predictions of the model, before the softmax.
    weights : `dictionary mapping str to mxnet.nd.array`
        The weights of the model.
    device : `mxnet.context.Context`
        The device, e.g. mxnet.cpu() or mxnet.gpu().
    num_classes : int
        The number of classes.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.

    """

    def __init__(
            self,
            data,
            logits,
            weights,
            device,
            num_classes,
            bounds,
            channel_axis=1):

        super(MXNetModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis)

        import mxnet as mx

        self._num_classes = num_classes

        self._device = device

        self._data_sym = data
        self._batch_logits_sym = logits

        label = mx.symbol.Variable('label')
        self._label_sym = label

        loss = mx.symbol.softmax_cross_entropy(logits, label)
        self._loss_sym = loss

        weight_names = list(weights.keys())
        weight_arrays = [weights[name] for name in weight_names]
        self._args_map = dict(zip(weight_names, weight_arrays))

    def num_classes(self):
        return self._num_classes

    def batch_predictions(self, images):
        import mxnet as mx
        data_array = mx.nd.array(images, ctx=self._device)
        self._args_map[self._data_sym.name] = data_array
        model = self._batch_logits_sym.bind(
            ctx=self._device, args=self._args_map, grad_req='null')
        model.forward(is_train=False)
        logits_array = model.outputs[0]
        logits = logits_array.asnumpy()
        return logits

    def predictions_and_gradient(self, image, label):
        import mxnet as mx
        label = np.asarray(label)
        data_array = mx.nd.array(image[np.newaxis], ctx=self._device)
        label_array = mx.nd.array(label[np.newaxis], ctx=self._device)
        self._args_map[self._data_sym.name] = data_array
        self._args_map[self._label_sym.name] = label_array

        grad_array = mx.nd.zeros(image[np.newaxis].shape, ctx=self._device)
        grad_map = {self._data_sym.name: grad_array}

        logits_loss = mx.sym.Group([self._batch_logits_sym, self._loss_sym])
        model = logits_loss.bind(
            ctx=self._device,
            args=self._args_map,
            args_grad=grad_map,
            grad_req='write')
        model.forward(is_train=True)
        logits_array = model.outputs[0]
        model.backward([
            mx.nd.zeros(logits_array.shape),
            mx.nd.array(np.array([1]))
        ])
        logits = logits_array.asnumpy()
        gradient = grad_array.asnumpy()
        return np.squeeze(logits, axis=0), np.squeeze(gradient, axis=0)
