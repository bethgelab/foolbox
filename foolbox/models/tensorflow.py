from __future__ import absolute_import

import numpy as np

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
            session = tf.Session(graph=images.graph)
            self._created_session = True
        else:
            self._created_session = False

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
            self._gradient = tf.squeeze(gradients[0], axis=0)

            self._bw_gradient_pre = tf.placeholder(tf.float32, self._logits.shape)  # noqa: E501
            bw_loss = tf.reduce_sum(self._logits * self._bw_gradient_pre)
            bw_gradients = tf.gradients(bw_loss, images)
            assert len(bw_gradients) == 1
            self._bw_gradient = tf.squeeze(bw_gradients[0], axis=0)

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
        images = self._process_input(images)
        predictions = self._session.run(
            self._batch_logits,
            feed_dict={self._images: images})
        return predictions

    def predictions_and_gradient(self, image, label):
        image = self._process_input(image)
        predictions, gradient = self._session.run(
            [self._logits, self._gradient],
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        gradient = self._process_gradient(gradient)
        return predictions, gradient

    def gradient(self, image, label):
        image = self._process_input(image)
        g = self._session.run(
            self._gradient,
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        g = self._process_gradient(g)
        return g

    def _loss_fn(self, image, label):
        image = self._process_input(image)
        loss = self._session.run(
            self._loss,
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        return loss

    def backward(self, gradient, image):
        assert gradient.ndim == 1
        image = self._process_input(image)
        g = self._session.run(
            self._bw_gradient,
            feed_dict={
                self._images: image[np.newaxis],
                self._bw_gradient_pre: gradient})
        g = self._process_gradient(g)
        assert g.shape == image.shape
        return g
