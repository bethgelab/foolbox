import pytest
import numpy as np

from foolbox.models import PyTorchModel


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_pytorch_model(num_classes):
    import torch
    import torch.nn as nn

    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            logits = x
            return logits

    model = Net()
    model = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes)

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


def test_pytorch_model_preprocessing():
    import torch
    import torch.nn as nn

    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            logits = x
            return logits

    model = Net()
    preprocessing = (np.arange(num_classes)[:, None, None],
                     np.random.uniform(size=(channels, 5, 5)) + 1)

    model1 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes)

    model2 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        preprocessing=preprocessing)

    model3 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes)

    np.random.seed(22)
    test_images = np.random.rand(2, channels, 5, 5).astype(np.float32)
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


def test_pytorch_model_gradient():
    import torch
    import torch.nn as nn

    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            logits = x
            return logits

    model = Net()
    preprocessing = (np.arange(num_classes)[:, None, None],
                     np.random.uniform(size=(channels, 5, 5)) + 1)

    model = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        preprocessing=preprocessing)

    epsilon = 1e-2

    np.random.seed(23)
    test_image = np.random.rand(channels, 5, 5).astype(np.float32)
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
def test_pytorch_backward(num_classes):
    import torch
    import torch.nn as nn

    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            logits = x
            return logits

    model = Net()
    model = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes)

    test_image = np.random.rand(channels, 5, 5).astype(np.float32)
    test_grad_pre = np.random.rand(num_classes).astype(np.float32)

    test_grad = model.backward(test_grad_pre, test_image)
    assert test_grad.shape == test_image.shape

    manual_grad = np.repeat(np.repeat(
        (test_grad_pre / 25.).reshape((-1, 1, 1)),
        5, axis=1), 5, axis=2)

    np.testing.assert_almost_equal(
        test_grad,
        manual_grad)
