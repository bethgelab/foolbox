import pytest
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
