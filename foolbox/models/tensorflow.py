from __future__ import absolute_import

import numpy as np
import logging

from .base import DifferentiableModel


class TensorFlowModel(DifferentiableModel):
    """Creates a :class:`Model` instance from existing `TensorFlow` tensors.

    Parameters
    ----------
    images : `tensorflow.Tensor`
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
            images,
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
            session = tf.Session(graph=images.graph)
            self._created_session = True
        else:
            self._created_session = False
            assert session.graph == images.graph, \
                'The default session uses the wrong graph'

        with session.graph.as_default():
            self._session = session
            self._images = images
            self._batch_logits = logits
            self._logits = tf.squeeze(logits, axis=0)
            self._label = tf.placeholder(tf.int64, (), name='label')

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._label[tf.newaxis],
                logits=self._logits[tf.newaxis])
            self._loss = tf.squeeze(loss, axis=0)
            gradients = tf.gradients(loss, images)
            assert len(gradients) == 1
            if gradients[0] is None:
                gradients[0] = tf.zeros_like(images)
            self._gradient = tf.squeeze(gradients[0], axis=0)

            self._bw_gradient_pre = tf.placeholder(tf.float32, self._logits.shape)  # noqa: E501
            bw_loss = tf.reduce_sum(self._logits * self._bw_gradient_pre)
            bw_gradients = tf.gradients(bw_loss, images)
            assert len(bw_gradients) == 1
            if bw_gradients[0] is None:
                bw_gradients[0] = tf.zeros_like(images)
            self._bw_gradient = tf.squeeze(bw_gradients[0], axis=0)

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
                raise ValueError(
                    'Please specify input_shape manually or '
                    'provide a model with an input_shape attribute')
        with tf.keras.backend.get_session().as_default():
            inputs = tf.placeholder(tf.float32, (None,) + input_shape)
            logits = model(inputs)
            return cls(inputs, logits, bounds=bounds,
                       channel_axis=channel_axis, preprocessing=preprocessing)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._created_session:
            self._session.close()
        return None

    @property
    def session(self):
        return self._session

    def num_classes(self):
        _, n = self._batch_logits.get_shape().as_list()
        return n

    def batch_predictions(self, images):
        images, _ = self._process_input(images)
        predictions = self._session.run(
            self._batch_logits,
            feed_dict={self._images: images})
        return predictions

    def predictions_and_gradient(self, image, label):
        image, dpdx = self._process_input(image)
        predictions, gradient = self._session.run(
            [self._logits, self._gradient],
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        gradient = self._process_gradient(dpdx, gradient)
        return predictions, gradient

    def gradient(self, image, label):
        image, dpdx = self._process_input(image)
        g = self._session.run(
            self._gradient,
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        g = self._process_gradient(dpdx, g)
        return g

    def _loss_fn(self, image, label):
        image, dpdx = self._process_input(image)
        loss = self._session.run(
            self._loss,
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        return loss

    def backward(self, gradient, image):
        assert gradient.ndim == 1
        input_shape = image.shape
        image, dpdx = self._process_input(image)
        g = self._session.run(
            self._bw_gradient,
            feed_dict={
                self._images: image[np.newaxis],
                self._bw_gradient_pre: gradient})
        g = self._process_gradient(dpdx, g)
        assert g.shape == input_shape
        return g
