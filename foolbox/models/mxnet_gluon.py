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
        self, block, bounds, num_classes, ctx=None, channel_axis=1, preprocessing=(0, 1)
    ):
        import mxnet as mx

        self._num_classes = num_classes

        if ctx is None:
            ctx = mx.cpu()

        super(MXNetGluonModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocessing=preprocessing
        )

        self._device = ctx
        self._block = block

    def num_classes(self):
        return self._num_classes

    def forward(self, inputs):
        import mxnet as mx

        inputs, _ = self._process_input(inputs)
        data_array = mx.nd.array(inputs, ctx=self._device)
        data_array.attach_grad()
        with mx.autograd.record(train_mode=False):
            L = self._block(data_array)
        return L.asnumpy()

    def forward_and_gradient_one(self, x, label):
        import mxnet as mx

        x, dpdx = self._process_input(x)
        label = mx.nd.array([label], ctx=self._device)
        data_array = mx.nd.array(x[np.newaxis], ctx=self._device)
        data_array.attach_grad()
        with mx.autograd.record(train_mode=False):
            logits = self._block(data_array)
            loss = mx.nd.softmax_cross_entropy(logits, label)
        loss.backward(train_mode=False)
        predictions = np.squeeze(logits.asnumpy(), axis=0)
        gradient = np.squeeze(data_array.grad.asnumpy(), axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        return predictions, gradient

    def forward_and_gradient(self, inputs, labels):
        import mxnet as mx

        inputs, dpdx = self._process_input(inputs)
        labels = mx.nd.array(labels, ctx=self._device)
        data_array = mx.nd.array(inputs, ctx=self._device)
        data_array.attach_grad()
        with mx.autograd.record(train_mode=False):
            logits = self._block(data_array)
            loss = mx.nd.softmax_cross_entropy(logits, labels)
        loss.backward(train_mode=False)
        predictions = logits.asnumpy()
        gradient = data_array.grad.asnumpy()
        gradient = self._process_gradient(dpdx, gradient)
        return predictions, gradient

    def gradient(self, inputs, labels):
        import mxnet as mx

        inputs, dpdx = self._process_input(inputs)
        inputs = mx.nd.array(inputs, ctx=self._device)
        labels = mx.nd.array(labels, ctx=self._device)
        inputs.attach_grad()
        with mx.autograd.record(train_mode=False):
            logits = self._block(inputs)
            loss = mx.nd.softmax_cross_entropy(logits, labels)
        loss.backward(train_mode=False)
        gradient = inputs.grad.asnumpy()
        gradient = self._process_gradient(dpdx, gradient)
        return gradient

    def _loss_fn(self, x, label):
        import mxnet as mx

        x, _ = self._process_input(x)

        label = np.array(label)

        if len(label.shape) == 0:
            # add batch dimension
            label = label[np.newaxis]
            x = x[np.newaxis]

        label = mx.nd.array(label, ctx=self._device)
        data_array = mx.nd.array(x, ctx=self._device)
        data_array.attach_grad()
        with mx.autograd.record(train_mode=False):
            logits = self._block(data_array)
            loss = mx.nd.softmax_cross_entropy(logits, label)
        loss.backward(train_mode=False)
        return loss.asnumpy()

    def backward(self, gradient, inputs):
        # lazy import
        import mxnet as mx

        assert gradient.ndim == 2
        inputs, dpdx = self._process_input(inputs)
        inputs = mx.nd.array(inputs, ctx=self._device)
        gradient = mx.nd.array(gradient, ctx=self._device)
        inputs.attach_grad()
        with mx.autograd.record(train_mode=False):
            logits = self._block(inputs)
        assert gradient.shape == logits.shape
        logits.backward(gradient, train_mode=False)
        gradient = inputs.grad.asnumpy()
        gradient = self._process_gradient(dpdx, gradient)
        return gradient
