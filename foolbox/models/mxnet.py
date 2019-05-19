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
            data,
            logits,
            args,
            ctx,
            num_classes,
            bounds,
            channel_axis=1,
            aux_states=None,
            preprocessing=(0, 1)):

        super(MXNetModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        import mxnet as mx

        self._num_classes = num_classes

        self._device = ctx

        self._data_sym = data
        self._logits_sym = logits

        labels = mx.symbol.Variable('labels')
        self._label_sym = labels

        # workaround for https://github.com/apache/incubator-mxnet/issues/6874
        # loss = mx.symbol.softmax_cross_entropy(logits, label)
        log_softmax = mx.sym.log_softmax(logits)
        loss = mx.sym.sum(
            mx.sym.one_hot(indices=labels, depth=num_classes) * log_softmax)
        loss = mx.sym.make_loss(loss)
        self._loss_sym = loss

        self._args_map = args.copy()
        self._aux_map = aux_states.copy() if aux_states is not None else None

        # move all parameters to correct device
        for k in self._args_map.keys():
            self._args_map[k] = \
                self._args_map[k].as_in_context(ctx)      # pragma: no cover

        if aux_states is not None:
            for k in self._aux_map.keys():                # pragma: no cover
                self._aux_map[k] = \
                    self._aux_map[k].as_in_context(ctx)   # pragma: no cover

    def num_classes(self):
        return self._num_classes

    def forward(self, inputs):
        import mxnet as mx
        inputs, _ = self._process_input(inputs)
        data_array = mx.nd.array(inputs, ctx=self._device)
        self._args_map[self._data_sym.name] = data_array
        model = self._logits_sym.bind(
            ctx=self._device, args=self._args_map, grad_req='null',
            aux_states=self._aux_map)
        model.forward(is_train=False)
        logits_array = model.outputs[0]
        logits = logits_array.asnumpy()
        return logits

    def forward_and_gradient_one(self, x, label):
        import mxnet as mx
        label = np.asarray(label)
        x, dpdx = self._process_input(x)
        data_array = mx.nd.array(x[np.newaxis], ctx=self._device)
        label_array = mx.nd.array(label[np.newaxis], ctx=self._device)
        self._args_map[self._data_sym.name] = data_array
        self._args_map[self._label_sym.name] = label_array

        grad_array = mx.nd.zeros(x[np.newaxis].shape, ctx=self._device)
        grad_map = {self._data_sym.name: grad_array}

        logits_loss = mx.sym.Group([self._logits_sym, self._loss_sym])
        model = logits_loss.bind(
            ctx=self._device,
            args=self._args_map,
            args_grad=grad_map,
            grad_req='write',
            aux_states=self._aux_map)
        model.forward(is_train=False)
        logits_array = model.outputs[0]
        model.backward([
            mx.nd.zeros(logits_array.shape),
            mx.nd.array(np.array([1]))
        ])
        logits = logits_array.asnumpy()
        gradient = grad_array.asnumpy()
        gradient = self._process_gradient(dpdx, gradient)
        return np.squeeze(logits, axis=0), np.squeeze(gradient, axis=0)

    def gradient(self, inputs, labels):
        import mxnet as mx
        inputs, dpdx = self._process_input(inputs)
        data_array = mx.nd.array(inputs, ctx=self._device)
        label_array = mx.nd.array(labels, ctx=self._device)
        self._args_map[self._data_sym.name] = data_array
        self._args_map[self._label_sym.name] = label_array

        grad_array = mx.nd.zeros(inputs.shape, ctx=self._device)
        grad_map = {self._data_sym.name: grad_array}

        model = self._loss_sym.bind(
            ctx=self._device,
            args=self._args_map,
            args_grad=grad_map,
            grad_req='write',
            aux_states=self._aux_map)
        model.forward(is_train=False)
        model.backward()
        gradient = grad_array.asnumpy()
        gradient = self._process_gradient(dpdx, gradient)
        return gradient

    def _loss_fn(self, x, label):
        import mxnet as mx
        x, _ = self._process_input(x)
        data_array = mx.nd.array(x[np.newaxis], ctx=self._device)
        label_array = mx.nd.array(np.array([label]), ctx=self._device)
        self._args_map[self._data_sym.name] = data_array
        self._args_map[self._label_sym.name] = label_array
        model = self._loss_sym.bind(
            ctx=self._device, args=self._args_map, grad_req='null',
            aux_states=self._aux_map)
        model.forward(is_train=False)
        loss_array = model.outputs[0]
        loss = loss_array.asnumpy()[0]
        return loss

    def backward(self, gradient, inputs):
        import mxnet as mx

        assert gradient.ndim == 2

        inputs, dpdx = self._process_input(inputs)
        data_array = mx.nd.array(inputs, ctx=self._device)
        self._args_map[self._data_sym.name] = data_array

        grad_array = mx.nd.zeros(inputs.shape, ctx=self._device)
        grad_map = {self._data_sym.name: grad_array}

        logits = self._logits_sym.bind(
            ctx=self._device,
            args=self._args_map,
            args_grad=grad_map,
            grad_req='write',
            aux_states=self._aux_map)

        logits.forward(is_train=False)

        gradient_pre_array = mx.nd.array(
            gradient, ctx=self._device)
        logits.backward(gradient_pre_array)

        gradient = grad_array.asnumpy()
        gradient = self._process_gradient(dpdx, gradient)
        return gradient
