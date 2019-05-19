from __future__ import absolute_import

import warnings
import numpy as np

from .base import DifferentiableModel


class TheanoModel(DifferentiableModel):
    """Creates a :class:`Model` instance from existing `Theano` tensors.

    Parameters
    ----------
    inputs : `theano.tensor`
        The input to the model.
    logits : `theano.tensor`
        The predictions of the model, before the softmax.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    num_classes : int
        Number of classes for which the model will output predictions.
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.

    """

    def __init__(
            self,
            inputs,
            logits,
            bounds,
            num_classes,
            channel_axis=1,
            preprocessing=[0, 1]):

        super(TheanoModel, self).__init__(bounds=bounds,
                                          channel_axis=channel_axis,
                                          preprocessing=preprocessing)

        warnings.warn('Theano is no longer being developed and Theano support'
                      ' in Foolbox will be removed', DeprecationWarning)

        self._num_classes = num_classes

        # delay import until class is instantiated
        import theano as th
        import theano.tensor as T

        labels = T.ivector('labels')
        loss = T.nnet.nnet.categorical_crossentropy(T.nnet.nnet.softmax(logits), labels).sum()
        gradient = th.gradient.grad(loss, inputs)

        backward_grad_logits = T.fmatrix('backward_grad_logits')
        backward_loss = (logits * backward_grad_logits).sum()
        backward_grad_inputs = th.gradient.grad(backward_loss, inputs)

        self._forward_fn = th.function([inputs], logits)
        self._forward_and_gradient_fn = th.function([inputs, labels], [logits, gradient])
        self._gradient_fn = th.function([inputs, labels], gradient)
        self._backward_fn = th.function([backward_grad_logits, inputs], backward_grad_inputs)

        # for tests
        self._loss_fn = th.function([inputs, labels], loss)

    def forward(self, inputs):
        inputs, _ = self._process_input(inputs)
        predictions = self._forward_fn(inputs)
        assert predictions.shape == (inputs.shape[0], self.num_classes())
        return predictions

    def forward_and_gradient_one(self, x, label):
        input_shape = x.shape
        x, dpdx = self._process_input(x)
        label = np.array(label, dtype=np.int32)
        predictions, gradient = self._forward_and_gradient_fn(x[np.newaxis], label[np.newaxis])
        gradient = gradient.astype(x.dtype)
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == input_shape
        assert gradient.dtype == x.dtype
        return predictions, gradient

    def _gradient_one(self, x, label):
        input_shape = x.shape
        x, dpdx = self._process_input(x)
        label = np.asarray(label, dtype=np.int32)
        gradient = self._gradient_fn(x[np.newaxis], label[np.newaxis])
        gradient = gradient.astype(x.dtype)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert gradient.shape == input_shape
        assert gradient.dtype == x.dtype
        return gradient

    def gradient(self, inputs, labels):
        if inputs.shape[0] == labels.shape[0] == 1:
            return self._gradient_one(inputs[0], labels[0])[np.newaxis]
        raise NotImplementedError

    def num_classes(self):
        return self._num_classes

    def _backward_one(self, gradient, x):
        assert gradient.ndim == 1
        input_shape = x.shape
        x, dpdx = self._process_input(x)
        gradient = self._backward_fn(gradient[np.newaxis], x[np.newaxis])
        gradient = gradient.astype(x.dtype)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert gradient.shape == input_shape
        assert gradient.dtype == x.dtype
        return gradient

    def backward(self, gradient, inputs):
        if inputs.shape[0] == gradient.shape[0] == 1:
            return self._backward_one(gradient[0], inputs[0])[np.newaxis]
        raise NotImplementedError
