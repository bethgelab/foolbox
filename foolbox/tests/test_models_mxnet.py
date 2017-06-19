import pytest
import mxnet as mx
import numpy as np

from foolbox.models import MXNetModel


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_model(num_classes):
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = mx.symbol.mean(images, axis=(2, 3))
        return logits

    images = mx.symbol.Variable('images')
    logits = mean_brightness_net(images)

    model = MXNetModel(
        images,
        logits,
        {},
        device=mx.cpu(),
        num_classes=num_classes,
        bounds=bounds,
        channel_axis=1)

    test_images = np.random.rand(2, channels, 5, 5).astype(np.float32)
    test_label = 7

    # Tests
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
