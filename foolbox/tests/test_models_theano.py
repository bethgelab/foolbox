import pytest
import numpy as np
import theano.tensor as T

from foolbox.models import TheanoModel


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_theano_model(num_classes):
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = T.mean(images, axis=(2, 3))
        return logits

    images = T.tensor4('images', dtype='float32')
    logits = mean_brightness_net(images)

    model = TheanoModel(
        images,
        logits,
        num_classes=num_classes,
        bounds=bounds)

    test_images = np.random.rand(2, channels, 5, 5).astype(np.float32)
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
@pytest.mark.parametrize('loss', [None, 'crossentropy', 'carlini'])
def test_theano_gradient(num_classes, loss):
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = T.mean(images, axis=(2, 3))
        return logits

    images = T.tensor4('images', dtype='float32')
    logits = mean_brightness_net(images)

    preprocessing = (np.arange(num_classes)[:, None, None],
                     np.random.uniform(size=(channels, 5, 5)) + 1)

    model = TheanoModel(
        images,
        logits,
        num_classes=num_classes,
        preprocessing=preprocessing,
        bounds=bounds)

    epsilon = 1e-2

    np.random.seed(23)
    test_image = np.random.rand(channels, 5, 5).astype(np.float32)
    test_label = 7

    _, g1 = model.predictions_and_gradient(test_image, test_label, loss=loss)

    test_image_p = model._process_input(test_image[None] - epsilon / 2 * g1)
    test_image_n = model._process_input(test_image[None] + epsilon / 2 * g1)
    l1 = model._loss_fn(test_image_p, [test_label], loss=loss)
    l2 = model._loss_fn(test_image_n, [test_label], loss=loss)

    # make sure that gradient is numerically correct
    np.testing.assert_array_almost_equal(
        1.,
        epsilon * np.linalg.norm(g1)**2 / (l2 - l1),
        decimal=1)

def test_theano_model_losses():
    num_classes = 3
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = T.mean(images, axis=(2, 3))
        return logits

    images = T.tensor4('images', dtype='float32')
    logits = mean_brightness_net(images)

    model = TheanoModel(
        images,
        logits,
        num_classes=num_classes,
        bounds=bounds)

    epsilon = 1e-2
    test_image = np.zeros((1, channels, 1, 1)).astype(np.float32)
    test_image[0, 0, 0, 0] = 1
    test_label = [0]

    logits = model.predictions(test_image[0])
    assert np.allclose(logits, [1, 0, 0])

    # test losses
    l0 = model._loss_fn(test_image, [0], loss=None)
    l1 = model._loss_fn(test_image, [1], loss=None)
    assert l0 < l1
    assert l0 == -1
    assert l1 == 0

    l0 = model._loss_fn(test_image, [0], loss='logits')
    l1 = model._loss_fn(test_image, [1], loss='logits')
    assert l0 < l1
    assert l0 == -1
    assert l1 == 0

    l0 = model._loss_fn(1e3 * test_image, [0], loss='crossentropy')
    l1 = model._loss_fn(1e3 * test_image, [1], loss='crossentropy')
    assert l0 < l1
    assert l0 == 0
    assert l1 == 1e3

    l0 = model._loss_fn(test_image, [0], loss='carlini')
    l1 = model._loss_fn(test_image, [1], loss='carlini')
    assert l0 < l1
    assert l0 == 0
    assert l1 == 1
