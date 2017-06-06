import tensorflow as tf
import numpy as np

from foolbox.models import TensorFlowModel


def test_tensorflow_model():
    for nc in [10, 1000]:
        g = tf.Graph()
        with g.as_default():
            images = tf.placeholder(tf.float32, (None, 224, 224, 3))
            logits = tf.tile(
                tf.reduce_mean(
                    images,
                    axis=(1, 2, 3)
                )[..., np.newaxis],
                [1, nc])

        with tf.Session(graph=g):
            model = TensorFlowModel(images, logits, bounds=(0, 255))

            _images = np.random.rand(2, 224, 224, 3).astype(np.float32)
            _label = 7

            assert model.batch_predictions(_images).shape == (2, nc)

            _logits = model.predictions(_images[0])
            assert _logits.shape == (nc,)

            _gradient = model.gradient(_images[0], _label)
            assert _gradient.shape == _images[0].shape

            np.testing.assert_almost_equal(
                model.predictions_and_gradient(_images[0], _label)[0],
                _logits)
            np.testing.assert_almost_equal(
                model.predictions_and_gradient(_images[0], _label)[1],
                _gradient)

            assert model.num_classes() == nc
