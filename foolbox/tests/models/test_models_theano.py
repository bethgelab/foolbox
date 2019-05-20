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

    images = T.tensor4('images')
    logits = mean_brightness_net(images)

    model = TheanoModel(
        images,
        logits,
        num_classes=num_classes,
        bounds=bounds)

    test_images = np.random.rand(2, channels, 5, 5).astype(np.float32)
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
def test_theano_gradient(num_classes):
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = T.mean(images, axis=(2, 3))
        return logits

    images = T.tensor4('images')
    logits = mean_brightness_net(images)

    preprocessing = (np.arange(num_classes)[:, None, None],
                     np.random.uniform(size=(channels, 5, 5)) + 1)

    model = TheanoModel(
        images,
        logits,
        num_classes=num_classes,
        preprocessing=preprocessing,
        bounds=bounds)

    # theano and lasagne calculate the cross-entropy from the probbilities
    # rather than combining softmax and cross-entropy calculation; they
    # therefore have lower numerical accuracy
    epsilon = 1e-3

    np.random.seed(23)
    test_image = np.random.rand(channels, 5, 5).astype(np.float32)
    test_label = 7

    _, g1 = model.forward_and_gradient_one(test_image, test_label)

    l1 = model._loss_fn(test_image[None] - epsilon / 2 * g1, [test_label])
    l2 = model._loss_fn(test_image[None] + epsilon / 2 * g1, [test_label])

    assert 1e5 * (l2 - l1) > 1

    # make sure that gradient is numerically correct
    np.testing.assert_array_almost_equal(
        1e5 * (l2 - l1),
        1e5 * epsilon * np.linalg.norm(g1)**2,
        decimal=1)


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_theano_backward(num_classes):
    bounds = (0, 255)
    channels = num_classes

    def mean_brightness_net(images):
        logits = T.mean(images, axis=(2, 3))
        return logits

    images = T.tensor4('images')
    logits = mean_brightness_net(images)

    model = TheanoModel(
        images,
        logits,
        num_classes=num_classes,
        bounds=bounds)

    test_image = np.random.rand(channels, 5, 5).astype(np.float32)
    test_grad_pre = np.random.rand(num_classes).astype(np.float32)

    test_grad = model.backward_one(test_grad_pre, test_image)
    assert test_grad.shape == test_image.shape

    manual_grad = np.repeat(np.repeat(
        (test_grad_pre / 25.).reshape((-1, 1, 1)),
        5, axis=1), 5, axis=2)

    np.testing.assert_almost_equal(
        test_grad,
        manual_grad)
