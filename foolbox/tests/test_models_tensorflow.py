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

        assert model.session is not None

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


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_tensorflow_preprocessing(num_classes):
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = tf.reduce_mean(images, axis=(1, 2))
        return logits

    q = (np.arange(num_classes)[None, None],
         np.random.uniform(size=(5, 5, channels)) + 1)

    g = tf.Graph()
    with g.as_default():
        images = tf.placeholder(tf.float32, (None, 5, 5, channels))
        logits = mean_brightness_net(images)

    with TensorFlowModel(images, logits, bounds=bounds,
                         preprocessing=q) as model:

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
def test_tensorflow_gradient(num_classes):
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = tf.reduce_mean(images, axis=(1, 2))
        return logits

    q = (np.arange(num_classes)[None, None],
         np.random.uniform(size=(5, 5, channels)) + 1)

    g = tf.Graph()
    with g.as_default():
        images = tf.placeholder(tf.float32, (None, 5, 5, channels))
        logits = mean_brightness_net(images)

    with TensorFlowModel(images, logits, bounds=bounds,
                         preprocessing=q) as model:

        epsilon = 1e-2

        np.random.seed(23)
        test_image = np.random.rand(5, 5, channels).astype(np.float32)
        test_label = 7

        _, g1 = model.predictions_and_gradient(test_image, test_label)

        l1 = model._loss_fn(test_image - epsilon / 2 * g1, test_label)
        l2 = model._loss_fn(test_image + epsilon / 2 * g1, test_label)

        assert 1e4 * (l2 - l1) > 1

        # make sure that gradient is numerically correct
        np.testing.assert_array_almost_equal(
            1e4 * (l2 - l1),
            1e4 * epsilon * np.linalg.norm(g1)**2,
            decimal=1)


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_tensorflow_backward(num_classes):
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

        assert model.session is not None

        test_image = np.random.rand(5, 5, channels).astype(np.float32)
        test_grad_pre = np.random.rand(num_classes).astype(np.float32)

        test_grad = model.backward(test_grad_pre, test_image)
        assert test_grad.shape == test_image.shape

        manual_grad = np.repeat(np.repeat(
            (test_grad_pre / 25.).reshape((1, 1, -1)),
            5, axis=0), 5, axis=1)

        np.testing.assert_almost_equal(
            test_grad,
            manual_grad)


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_tensorflow_model_non_diff(num_classes):
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = tf.reduce_mean(images, axis=(1, 2))
        return logits

    g = tf.Graph()
    with g.as_default():
        images = tf.placeholder(tf.float32, (None, 5, 5, channels))
        images_nd = tf.cast(images > 0, tf.float32)
        logits = mean_brightness_net(images_nd)

    with tf.Session(graph=g):
        model = TensorFlowModel(
            images,
            logits,
            bounds=bounds)

        assert model.session is not None

        test_images = np.random.rand(5, 5, channels).astype(np.float32)
        test_label = 7

        test_gradient = model.gradient(test_images, test_label)
        assert (test_gradient == 0).all()
