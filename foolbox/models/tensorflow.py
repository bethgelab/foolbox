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
            self._label = tf.placeholder(tf.int32, (), name='label')

        self._loss_cache = {}
        self._grad_cache = {}

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

    def _loss(self, loss, **kwargs):
        import tensorflow as tf
        try:
            return self._loss_cache[loss]
        except KeyError:
            if hasattr(loss, '__call__'):
                return loss(self._logits, self._label)
            elif loss in [None, 'logits']:
                with self.session.graph.as_default():
                    sym_loss = -self._logits[self._label]
            elif loss == 'crossentropy':
                with self.session.graph.as_default():
                    sym_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self._label[tf.newaxis],
                        logits=self._logits[tf.newaxis])
                    sym_loss = tf.squeeze(sym_loss, axis=0)
            elif loss == 'carlini':
                with self.session.graph.as_default():
                    sym_loss = tf.reduce_max(self._logits, axis=0) \
                           - self._logits[self._label]
                    sym_loss = tf.nn.relu(sym_loss)
            else:
                raise NotImplementedError('The loss {} is currently not \
                        implemented for this framework.'.format(loss))

            self._loss_cache[loss] = sym_loss
            return sym_loss

    def _gradient(self, loss, **kwargs):
        import tensorflow as tf
        try:
            return self._grad_cache[loss]
        except KeyError:
            with self.session.graph.as_default():
                gradients = tf.gradients(self._loss(loss), self._images)
                assert len(gradients) == 1
                self._grad_cache[loss] = tf.squeeze(gradients[0], axis=0)
                return self._grad_cache[loss]

    def batch_predictions(self, images):
        images = self._process_input(images)
        predictions = self._session.run(
            self._batch_logits,
            feed_dict={self._images: images})
        return predictions

    def predictions_and_gradient(self, image, label, loss=None, **kwargs):
        image = self._process_input(image)
        predictions, gradient = self._session.run(
            [self._logits, self._gradient(loss, **kwargs)],
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        gradient = self._process_gradient(gradient)
        return predictions, gradient

    def gradient(self, image, label, loss=None, **kwargs):
        image = self._process_input(image)
        g = self._session.run(
            self._gradient(loss, **kwargs),
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        g = self._process_gradient(g)
        return g

    def _loss_fn(self, image, label, loss=None):
        image = self._process_input(image)
        loss = self._session.run(
            self._loss(loss),
            feed_dict={
                self._images: image[np.newaxis],
                self._label: label})
        return loss
