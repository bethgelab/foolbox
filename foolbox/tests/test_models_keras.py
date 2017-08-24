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

    p1 = model1.batch_predictions(test_images)
    p2 = model2.batch_predictions(test_images)
    p3 = model3.batch_predictions(test_images)

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

    np.random.seed(22)
    test_images = np.random.rand(2, 5, 5, channels).astype(np.float32)
    test_images_copy = test_images.copy()

    p1 = model1.batch_predictions(test_images)
    p2 = model2.batch_predictions(test_images)

    # make sure the images have not been changed by
    # the in-place preprocessing
    assert np.all(test_images == test_images_copy)

    p3 = model3.batch_predictions(test_images)

    assert p1.shape == p2.shape == p3.shape == (2, num_classes)

    np.testing.assert_array_almost_equal(
        p1 - p1.max(),
        p3 - p3.max(),
        decimal=5)

@pytest.mark.parametrize('loss', [None, 'crossentropy', 'carlini'])
def test_keras_model_gradients(loss):
    num_classes = 10
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

    p1, g1 = model.predictions_and_gradient(test_image, test_label, loss=loss)

    l1 = model._loss_fn(test_image - eps / 2 * g1, test_label, loss=loss)
    l2 = model._loss_fn(test_image + eps / 2 * g1, test_label, loss=loss)

    # make sure that gradient is numerically correct
    np.testing.assert_array_almost_equal(
        1.,
        eps * np.linalg.norm(g1)**2 / (l2 - l1),
        decimal=1)

def test_keras_model_losses():
    num_classes = 3
    bounds = (0, 255)
    channels = num_classes

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        inputs = Input(shape=(1, 1, channels))
        logits = GlobalAveragePooling2D(
            data_format='channels_last')(inputs)

        model = KerasModel(
            Model(inputs=inputs, outputs=logits),
            bounds=bounds,
            predicts='logits')

    epsilon = 1e-2
    test_image = np.zeros((1, 1, channels)).astype(np.float32)
    test_image[0, 0, 0] = 1
    test_label = 0

    logits = model.predictions(test_image)
    assert np.allclose(logits, [1, 0, 0])

    # test losses
    l0 = model._loss_fn(test_image, 0, loss=None)
    l1 = model._loss_fn(test_image, 1, loss=None)
    assert l0 < l1
    assert l0 == -1
    assert l1 == 0

    l0 = model._loss_fn(test_image, 0, loss='logits')
    l1 = model._loss_fn(test_image, 1, loss='logits')
    assert l0 < l1
    assert l0 == -1
    assert l1 == 0

    l0 = model._loss_fn(1e3 * test_image, 0, loss='crossentropy')
    l1 = model._loss_fn(1e3 * test_image, 1, loss='crossentropy')
    assert l0 < l1
    assert l0 == 0
    assert l1 == 1e3

    l0 = model._loss_fn(test_image, 0, loss='carlini')
    l1 = model._loss_fn(test_image, 1, loss='carlini')
    assert l0 < l1
    assert l0 == 0
    assert l1 == 1
