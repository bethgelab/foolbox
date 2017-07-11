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

    def preprocess_fn(x):
        # modify x in-place
        x /= 2
        return x

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
        preprocess_fn=preprocess_fn)

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
