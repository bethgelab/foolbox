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

        assert model.forward(test_images).shape \
            == (2, num_classes)

        test_logits = model.forward_one(test_images[0])
        assert test_logits.shape == (num_classes,)

        test_gradient = model.gradient_one(test_images[0], test_label)
        assert test_gradient.shape == test_images[0].shape

        np.testing.assert_almost_equal(
            model.forward_and_gradient_one(test_images[0], test_label)[0],
            test_logits)
        np.testing.assert_almost_equal(
            model.forward_and_gradient_one(test_images[0], test_label)[1],
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

        assert model.forward(test_images).shape \
            == (2, num_classes)

        test_logits = model.forward_one(test_images[0])
        assert test_logits.shape == (num_classes,)

        test_gradient = model.gradient_one(test_images[0], test_label)
        assert test_gradient.shape == test_images[0].shape

        np.testing.assert_almost_equal(
            model.forward_and_gradient_one(test_images[0], test_label)[0],
            test_logits)
        np.testing.assert_almost_equal(
            model.forward_and_gradient_one(test_images[0], test_label)[1],
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

        assert model.forward(test_images).shape \
            == (2, num_classes)

        test_logits = model.forward_one(test_images[0])
        assert test_logits.shape == (num_classes,)

        test_gradient = model.gradient_one(test_images[0], test_label)
        assert test_gradient.shape == test_images[0].shape

        np.testing.assert_almost_equal(
            model.forward_and_gradient_one(test_images[0], test_label)[0],
            test_logits)
        np.testing.assert_almost_equal(
            model.forward_and_gradient_one(test_images[0], test_label)[1],
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

        _, g1 = model.forward_and_gradient_one(test_image, test_label)

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

        test_grad = model.backward_one(test_grad_pre, test_image)
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

        test_image = np.random.rand(5, 5, channels).astype(np.float32)
        test_label = 7

        test_gradient = model.gradient_one(test_image, test_label)
        assert (test_gradient == 0).all()


def test_tf_keras_constructor():
    bounds = (0, 255)

    def create_model():
        data_format = 'channels_last'
        input_shape = [28, 28, 1]
        l = tf.keras.layers  # noqa: E741
        max_pool = l.MaxPooling2D(
            (2, 2), (2, 2), padding='same', data_format=data_format)
        return tf.keras.Sequential(
            [
                l.Conv2D(
                    32,
                    5,
                    padding='same',
                    data_format=data_format,
                    input_shape=input_shape,
                    activation=tf.nn.relu),
                max_pool,
                l.Conv2D(
                    64,
                    5,
                    padding='same',
                    data_format=data_format,
                    activation=tf.nn.relu),
                max_pool,
                l.Flatten(),
                l.Dense(1024, activation=tf.nn.relu),
                l.Dropout(0.4),
                l.Dense(10)
            ])
    model = create_model()
    fmodel = TensorFlowModel.from_keras(model, bounds=bounds)
    assert fmodel.num_classes() == 10

    fmodel.session.run(tf.global_variables_initializer())

    test_images = np.random.rand(2, 28, 28, 1).astype(np.float32)
    assert fmodel.forward(test_images).shape == (2, 10)


def test_tf_keras_exception():
    bounds = (0, 255)

    def mean_brightness_net(images):
        logits = tf.reduce_mean(images, axis=(1, 2))
        return logits

    model = mean_brightness_net
    with pytest.raises(ValueError):
        TensorFlowModel.from_keras(model, bounds=bounds)

    TensorFlowModel.from_keras(model, bounds=bounds, input_shape=(5, 5, 3))
