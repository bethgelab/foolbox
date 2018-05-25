import pytest
import mxnet as mx
import numpy as np

from foolbox.models import MXNetGluonModel
from mxnet.gluon import HybridBlock


class MeanBrightnessNet(HybridBlock):
    def hybrid_forward(self, F, x, *args, **kwargs):
        return mx.nd.mean(x, axis=(2, 3))


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_model(num_classes):
    bounds = (0, 255)
    channels = num_classes

    block = MeanBrightnessNet()

    model = MXNetGluonModel(
        block,
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


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_model_gradient(num_classes):
    bounds = (0, 255)
    channels = num_classes

    block = MeanBrightnessNet()

    model = MXNetGluonModel(
        block,
        ctx=mx.cpu(),
        num_classes=num_classes,
        bounds=bounds,
        channel_axis=1)

    test_images = np.random.rand(2, channels, 5, 5).astype(np.float32)
    test_image = test_images[0]
    test_label = 7

    epsilon = 1e-2
    _, g1 = model.predictions_and_gradient(test_image, test_label)
    l1 = model._loss_fn(test_image - epsilon / 2 * g1, test_label)
    l2 = model._loss_fn(test_image + epsilon / 2 * g1, test_label)

    assert 1e4 * (l2 - l1) > 1

    # make sure that gradient is numerically correct
    np.testing.assert_array_almost_equal(
        1e4 * (l2 - l1),
        1e4 * epsilon * np.linalg.norm(g1)**2,
        decimal=1)
