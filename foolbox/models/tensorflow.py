from __future__ import absolute_import

import numpy as np
import logging

from .base import DifferentiableModel


class TensorFlowModel(DifferentiableModel):
    """Creates a :class:`Model` instance from existing `TensorFlow` tensors.

    Parameters
    ----------
    inputs : `tensorflow.Tensor`
        The input to the model, usually a `tensorflow.placeholder`.
    logits : `tensorflow.Tensor`
        The predictions of the model, before the softmax.
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
            inputs,
            logits,
            bounds,
            channel_axis=3,
            preprocessing=(0, 1)):

        super(TensorFlowModel, self).__init__(bounds=bounds,
                                              channel_axis=channel_axis,
                                              preprocessing=preprocessing)

        # delay import until class is instantiated
        import tensorflow as tf

        session = tf.get_default_session()
        if session is None:
            logging.warning('No default session. Created a new tf.Session. '
                            'Please restore variables using this session.')
            session = tf.Session(graph=inputs.graph)
            self._created_session = True
        else:
            self._created_session = False
            assert session.graph == inputs.graph, \
                'The default session uses the wrong graph'

        with session.graph.as_default():
            self._session = session
            self._inputs = inputs
            self._logits = logits

            labels = tf.placeholder(tf.int64, (None,), name='labels')
            self._labels = labels

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_sum(loss)
            self._loss = loss

            gradient, = tf.gradients(loss, inputs)
            if gradient is None:
                gradient = tf.zeros_like(inputs)
            self._gradient = gradient

            backward_grad_logits = tf.placeholder(tf.float32, logits.shape)
            backward_loss = tf.reduce_sum(logits * backward_grad_logits)
            backward_grad_inputs, = tf.gradients(backward_loss, inputs)
            if backward_grad_inputs is None:
                backward_grad_inputs = tf.zeros_like(inputs)

            self._backward_grad_logits = backward_grad_logits
            self._backward_grad_inputs = backward_grad_inputs

    @classmethod
    def from_keras(cls, model, bounds, input_shape=None,
                   channel_axis=3, preprocessing=(0, 1)):
        """Alternative constructor for a TensorFlowModel that
        accepts a `tf.keras.Model` instance.

        Parameters
        ----------
        model : `tensorflow.keras.Model`
            A `tensorflow.keras.Model` that accepts a single input tensor
            and returns a single output tensor representing logits.
        bounds : tuple
            Tuple of lower and upper bound for the pixel values, usually
            (0, 1) or (0, 255).
        input_shape : tuple
            The shape of a single input, e.g. (28, 28, 1) for MNIST.
            If None, tries to get the the shape from the model's
            input_shape attribute.
        channel_axis : int
            The index of the axis that represents color channels.
        preprocessing: 2-element tuple with floats or numpy arrays
            Elementwises preprocessing of input; we first subtract the first
            element of preprocessing from the input and then divide the input
            by the second element.

        """
        import tensorflow as tf
        if input_shape is None:
            try:
                input_shape = model.input_shape[1:]
            except AttributeError:
                raise ValueError('Please specify input_shape manually or provide a model with an input_shape attribute')
        with tf.keras.backend.get_session().as_default():
            inputs = tf.placeholder(tf.float32, (None,) + input_shape)
            logits = model(inputs)
            return cls(inputs, logits, bounds=bounds, channel_axis=channel_axis, preprocessing=preprocessing)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._created_session:
            self._session.close()
        return None

    @property
    def session(self):
        return self._session

    def num_classes(self):
        _, n = self._logits.get_shape().as_list()
        return n

    def forward(self, inputs):
        inputs, _ = self._process_input(inputs)
        predictions = self._session.run(self._logits, feed_dict={self._inputs: inputs})
        return predictions

    def forward_and_gradient_one(self, x, label):
        x, dpdx = self._process_input(x)
        predictions, gradient = self._session.run(
            [self._logits, self._gradient],
            feed_dict={self._inputs: x[np.newaxis], self._labels: np.asarray(label)[np.newaxis]})
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        return predictions, gradient

    def gradient(self, inputs, labels):
        inputs, dpdx = self._process_input(inputs)
        g = self._session.run(
            self._gradient,
            feed_dict={
                self._inputs: inputs,
                self._labels: labels})
        g = self._process_gradient(dpdx, g)
        return g

    def _loss_fn(self, x, label):
        x, dpdx = self._process_input(x)
        loss = self._session.run(
            self._loss,
            feed_dict={
                self._inputs: x[np.newaxis],
                self._labels: np.asarray(label)[np.newaxis]})
        return loss

    def backward(self, gradient, inputs):
        assert gradient.ndim == 2
        input_shape = inputs.shape
        inputs, dpdx = self._process_input(inputs)
        g = self._session.run(
            self._backward_grad_inputs,
            feed_dict={
                self._inputs: inputs,
                self._backward_grad_logits: gradient})
        g = self._process_gradient(dpdx, g)
        assert g.shape == input_shape
        return g
