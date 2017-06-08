import pytest
import tensorflow as tf
import numpy as np

from foolbox.models import TensorFlowModel


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_tensorflow_model(num_classes):
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = tf.reduce_mean(images, axis=(1, 2))
        return logits

    g = tf.Graph()
    with g.as_default():
        images = tf.placeholder(tf.float32, (None, 5, 5, channels))
        logits = mean_brightness_net(images)

    with tf.Session(graph=g):
        model = TensorFlowModel(
            images,
            logits,
            bounds=bounds)

        test_images = np.random.rand(2, 5, 5, channels).astype(np.float32)
        test_label = 7

        assert model.batch_predictions(test_images).shape \
            == (2, num_classes)

        test_logits = model.predictions(test_images[0])
        assert test_logits.shape == (num_classes,)

        test_gradient = model.gradient(test_images[0], test_label)
        assert test_gradient.shape == test_images[0].shape

        np.testing.assert_almost_equal(
            model.predictions_and_gradient(test_images[0], test_label)[0],
            test_logits)
        np.testing.assert_almost_equal(
            model.predictions_and_gradient(test_images[0], test_label)[1],
            test_gradient)

        assert model.num_classes() == num_classes


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_tensorflow_model_cm(num_classes):
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = tf.reduce_mean(images, axis=(1, 2))
        return logits

    g = tf.Graph()
    with g.as_default():
        images = tf.placeholder(tf.float32, (None, 5, 5, channels))
        logits = mean_brightness_net(images)

    with TensorFlowModel(images, logits, bounds=bounds) as model:

        test_images = np.random.rand(2, 5, 5, channels).astype(np.float32)
        test_label = 7

        assert model.batch_predictions(test_images).shape \
            == (2, num_classes)

        test_logits = model.predictions(test_images[0])
        assert test_logits.shape == (num_classes,)

        test_gradient = model.gradient(test_images[0], test_label)
        assert test_gradient.shape == test_images[0].shape

        np.testing.assert_almost_equal(
            model.predictions_and_gradient(test_images[0], test_label)[0],
            test_logits)
        np.testing.assert_almost_equal(
            model.predictions_and_gradient(test_images[0], test_label)[1],
            test_gradient)

        assert model.num_classes() == num_classes
