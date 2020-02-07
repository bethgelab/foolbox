import pytest
import numpy as np

from foolbox.models import JAXModel


@pytest.mark.parametrize("num_classes", [10, 1000])
def test_jax_model(num_classes):
    import jax.numpy as jnp

    bounds = (0, 255)
    channels = num_classes

    def net(x):
        return jnp.mean(x, axis=(1, 2))

    model = JAXModel(net, bounds=bounds, num_classes=num_classes)

    test_images = np.random.rand(2, 5, 5, channels).astype(np.float32)
    test_label = 7

    assert model.forward(test_images).shape == (2, num_classes)

    test_logits = model.forward_one(test_images[0])
    assert test_logits.shape == (num_classes,)

    test_gradient = model.gradient_one(test_images[0], test_label)
    assert test_gradient.shape == test_images[0].shape

    np.testing.assert_almost_equal(
        model.forward_and_gradient_one(test_images[0], test_label)[0], test_logits
    )
    np.testing.assert_almost_equal(
        model.forward_and_gradient_one(test_images[0], test_label)[1], test_gradient
    )

    assert model.num_classes() == num_classes


def test_jax_model_preprocessing():
    import jax.numpy as jnp

    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    def net(x):
        return jnp.mean(x, axis=(1, 2))

    model = net

    q = (
        np.arange(num_classes)[None, None],
        np.random.uniform(size=(5, 5, channels)) + 1,
    )

    model1 = JAXModel(model, bounds=bounds, num_classes=num_classes)

    model2 = JAXModel(model, bounds=bounds, num_classes=num_classes, preprocessing=q)

    model3 = JAXModel(model, bounds=bounds, num_classes=num_classes)

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

    np.testing.assert_array_almost_equal(p1 - p1.max(), p3 - p3.max(), decimal=5)


def test_jax_model_gradient():
    import jax.numpy as jnp

    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    def net(x):
        return jnp.mean(x, axis=(1, 2))

    q = (
        np.arange(num_classes)[None, None],
        np.random.uniform(size=(5, 5, channels)) + 1,
    )

    model = JAXModel(net, bounds=bounds, num_classes=num_classes, preprocessing=q)

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
        1e4 * (l2 - l1), 1e4 * epsilon * np.linalg.norm(g1) ** 2, decimal=1
    )


def test_jax_model_forward_gradient():
    import jax.numpy as jnp

    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    def net(x):
        return jnp.mean(x, axis=(1, 2))

    q = (
        np.arange(num_classes)[None, None],
        np.random.uniform(size=(5, 5, channels)) + 1,
    )

    model = JAXModel(net, bounds=bounds, num_classes=num_classes, preprocessing=q)

    epsilon = 1e-2

    np.random.seed(23)
    test_images = np.random.rand(5, 5, 5, channels).astype(np.float32)
    test_labels = [7] * 5

    _, g1 = model.forward_and_gradient(test_images, test_labels)

    l1 = model._loss_fn(test_images - epsilon / 2 * g1, test_labels)
    l2 = model._loss_fn(test_images + epsilon / 2 * g1, test_labels)

    assert 1e5 * (l2 - l1) > 1

    # make sure that gradient is numerically correct
    np.testing.assert_array_almost_equal(
        1e5 * (l2 - l1), 1e5 * epsilon * np.linalg.norm(g1) ** 2, decimal=1
    )


@pytest.mark.parametrize("num_classes", [10, 1000])
def test_jax_backward(num_classes):
    import jax.numpy as jnp

    bounds = (0, 255)
    channels = num_classes

    def net(x):
        return jnp.mean(x, axis=(1, 2))

    model = JAXModel(net, bounds=bounds, num_classes=num_classes)

    test_image = np.random.rand(5, 5, channels).astype(np.float32)
    test_grad_pre = np.random.rand(num_classes).astype(np.float32)

    test_grad = model.backward_one(test_grad_pre, test_image)
    assert test_grad.shape == test_image.shape

    manual_grad = np.repeat(
        np.repeat((test_grad_pre / 25.0).reshape((1, 1, -1)), 5, axis=0), 5, axis=1
    )

    np.testing.assert_almost_equal(test_grad, manual_grad)
