import pytest
import warnings

import numpy as np
from keras.layers import GlobalAveragePooling2D
from keras.layers import Activation
from keras.layers import Input
from keras.activations import softmax
from keras.models import Model
from keras.models import Sequential

from foolbox.models import KerasModel


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_keras_model(num_classes):

    bounds = (0, 255)
    channels = num_classes

    model = Sequential()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model.add(GlobalAveragePooling2D(
            data_format='channels_last', input_shape=(5, 5, channels)))

        model = KerasModel(
            model,
            bounds=bounds,
            predicts='logits')

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
def test_keras_model_probs(num_classes):
    bounds = (0, 255)
    channels = num_classes

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        inputs = Input(shape=(5, 5, channels))
        logits = GlobalAveragePooling2D(
            data_format='channels_last')(inputs)
        probs = Activation(softmax)(logits)

        model1 = KerasModel(
            Model(inputs=inputs, outputs=logits),
            bounds=bounds,
            predicts='logits')

        model2 = KerasModel(
            Model(inputs=inputs, outputs=probs),
            bounds=bounds,
            predicts='probabilities')

        model3 = KerasModel(
            Model(inputs=inputs, outputs=probs),
            bounds=bounds,
            predicts='probs')

    np.random.seed(22)
    test_images = np.random.rand(2, 5, 5, channels).astype(np.float32)

    p1 = model1.forward(test_images)
    p2 = model2.forward(test_images)
    p3 = model3.forward(test_images)

    assert p1.shape == p2.shape == p3.shape == (2, num_classes)

    np.testing.assert_array_almost_equal(
        p1 - p1.max(),
        p2 - p2.max(),
        decimal=1)

    np.testing.assert_array_almost_equal(
        p2 - p2.max(),
        p3 - p3.max(),
        decimal=5)


def test_keras_model_preprocess():
    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        inputs = Input(shape=(5, 5, channels))
        logits = GlobalAveragePooling2D(
            data_format='channels_last')(inputs)

        preprocessing = (np.arange(num_classes)[None, None],
                         np.random.uniform(size=(5, 5, channels)) + 1)

        model1 = KerasModel(
            Model(inputs=inputs, outputs=logits),
            bounds=bounds,
            predicts='logits')

        model2 = KerasModel(
            Model(inputs=inputs, outputs=logits),
            bounds=bounds,
            predicts='logits',
            preprocessing=preprocessing)

        model3 = KerasModel(
            Model(inputs=inputs, outputs=logits),
            bounds=bounds,
            predicts='logits')

        preprocessing = (0, np.random.uniform(size=(5, 5, channels)) + 1)

        model4 = KerasModel(
            Model(inputs=inputs, outputs=logits),
            bounds=bounds,
            predicts='logits',
            preprocessing=preprocessing)

    np.random.seed(22)
    test_images = np.random.rand(2, 5, 5, channels).astype(np.float32)
    test_images_copy = test_images.copy()

    p1 = model1.forward(test_images)
    p2 = model2.forward(test_images)

    # make sure the images have not been changed by
    # the in-place preprocessing
    assert np.all(test_images == test_images_copy)

    p3 = model3.forward(test_images)

    assert p1.shape == p2.shape == p3.shape == (2, num_classes)

    np.testing.assert_array_almost_equal(
        p1 - p1.max(),
        p3 - p3.max(),
        decimal=5)

    model4.forward(test_images)


def test_keras_model_gradients():
    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        inputs = Input(shape=(5, 5, channels))
        logits = GlobalAveragePooling2D(
            data_format='channels_last')(inputs)

        preprocessing = (np.arange(num_classes)[None, None],
                         np.random.uniform(size=(5, 5, channels)) + 1)

        model = KerasModel(
            Model(inputs=inputs, outputs=logits),
            bounds=bounds,
            predicts='logits',
            preprocessing=preprocessing)

    eps = 1e-3

    np.random.seed(22)
    test_image = np.random.rand(5, 5, channels).astype(np.float32)
    test_label = 7

    _, g1 = model.forward_and_gradient_one(test_image, test_label)

    test_label_array = np.array([test_label])
    l1 = model._loss_fn([test_image[None] - eps / 2 * g1, test_label_array])[0]
    l2 = model._loss_fn([test_image[None] + eps / 2 * g1, test_label_array])[0]

    assert 1e5 * (l2 - l1) > 1

    # make sure that gradient is numerically correct
    np.testing.assert_array_almost_equal(
        1e5 * (l2 - l1),
        1e5 * eps * np.linalg.norm(g1)**2,
        decimal=1)


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_keras_backward(num_classes):

    bounds = (0, 255)
    channels = num_classes

    model = Sequential()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model.add(GlobalAveragePooling2D(
            data_format='channels_last', input_shape=(5, 5, channels)))

        model = KerasModel(
            model,
            bounds=bounds,
            predicts='logits')

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
