import pytest
import numpy as np
import torch
import torch.nn as nn

from foolbox.models import PyTorchModel


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_pytorch_model(num_classes):

    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.squeeze(x, dim=3)
            x = torch.mean(x, 2)
            x = torch.squeeze(x, dim=2)
            logits = x
            return logits

    model = Net()
    model = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        cuda=False)

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
    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.squeeze(x, dim=3)
            x = torch.mean(x, 2)
            x = torch.squeeze(x, dim=2)
            logits = x
            return logits

    model = Net()
    preprocessing = (np.arange(num_classes)[:, None, None],
                     np.random.uniform(size=(channels, 5, 5)) + 1)

    model1 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        cuda=False)

    model2 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        cuda=False,
        preprocessing=preprocessing)

    model3 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        cuda=False)

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

@pytest.mark.parametrize('loss', [None, 'crossentropy', 'carlini'])
def test_pytorch_model_gradient(loss):
    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.squeeze(x, dim=3)
            x = torch.mean(x, 2)
            x = torch.squeeze(x, dim=2)
            logits = x
            return logits

    model = Net()
    preprocessing = (np.arange(num_classes)[:, None, None],
                     np.random.uniform(size=(channels, 5, 5)) + 1)

    model = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        cuda=False,
        preprocessing=preprocessing)

    epsilon = 1e-2

    np.random.seed(23)
    test_image = np.random.rand(channels, 5, 5).astype(np.float32)
    test_label = 7

    _, g1 = model.predictions_and_gradient(test_image, test_label, loss=loss)

    l1 = model._loss_fn(test_image - epsilon / 2 * g1, test_label, loss=loss)
    l2 = model._loss_fn(test_image + epsilon / 2 * g1, test_label, loss=loss)

    # make sure that gradient is numerically correct
    np.testing.assert_array_almost_equal(
        1.,
        epsilon * np.linalg.norm(g1)**2 / (l2 - l1),
        decimal=1)

def test_pytorch_model_losses():
    num_classes = 3
    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.squeeze(x, dim=3)
            x = torch.mean(x, 2)
            x = torch.squeeze(x, dim=2)
            logits = x
            return logits

    model = Net()

    model = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        cuda=False)

    epsilon = 1e-2
    test_image = np.zeros((channels, 1, 1)).astype(np.float32)
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
