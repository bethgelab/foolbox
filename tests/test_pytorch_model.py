import numpy as np
import torch
import torch.nn as nn

from foolbox.ext.native.models import PyTorchModel


def test_pytorch_model():
    channels = num_classes = 10
    batch_size = 8
    h = w = 32
    bounds = (0, 1)

    class Model(nn.Module):
        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            return x

    model = Model()
    fmodel = PyTorchModel(model, bounds=bounds)

    np.random.seed(0)
    x = np.random.uniform(*bounds, size=(batch_size, channels, h, w)).astype(np.float32)
    x = torch.from_numpy(x).to(fmodel.device)
    y = np.arange(batch_size) % num_classes
    y = torch.from_numpy(y).to(fmodel.device)

    output = fmodel.forward(x)
    assert output.shape == (batch_size, num_classes)
    assert isinstance(output, torch.Tensor)

    output = fmodel.gradient(x, y)
    assert output.shape == x.shape
    assert isinstance(output, torch.Tensor)
