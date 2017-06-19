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

    """

    def __init__(
            self,
            images,
            logits,
            bounds,
            channel_axis=3):

        super(TensorFlowModel, self).__init__(bounds=bounds,
                                              channel_axis=channel_axis)

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
            loss = tf.squeeze(loss, axis=0)
            gradients = tf.gradients(loss, images)
            assert len(gradients) == 1
            self._gradient = tf.squeeze(gradients[0], axis=0)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._created_session:
            self._session.close()
        return None

    def num_classes(self):
        _, n = self._batch_logits.get_shape().as_list()
        return n

    def batch_predictions(self, images):
        predictions = self._session.run(
            self._batch_logits,
            feed_dict={self._images: images})
        return predictions

    def predictions_and_gradient(self, image, label):
        predictions, gradient = self._session.run(
            [self._logits, self._gradient],
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        return predictions, gradient

    def gradient(self, image, label):
        g = self._session.run(
            self._gradient,
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        return g
