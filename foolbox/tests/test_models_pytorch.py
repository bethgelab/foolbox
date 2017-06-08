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
            super().__init__()

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
