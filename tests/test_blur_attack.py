import eagerpy as ep
import numpy as np
import torch
import torch.nn as nn

from foolbox.ext.native.devutils import flatten
from foolbox.ext.native.models import PyTorchModel
from foolbox.ext.native.attacks import GaussianBlurAttack


def test_gaussian_blur_attack():
    channels = 3
    batch_size = 8
    h = w = 32
    bounds = (0, 1)

    class Model(nn.Module):
        def forward(self, x):
            # instead of our usual model that's robust to the BlurAttack,
            # we use a slighlty different model that can be attacked
            x = x[:, :, 1:, :] - x[:, :, :-1, :]
            x = x[:, :, :, 1:] - x[:, :, :, :-1]
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            return x

    model = Model().eval()
    fmodel = PyTorchModel(model, bounds=bounds)

    np.random.seed(0)
    x = np.random.uniform(*bounds, size=(batch_size, channels, h, w)).astype(np.float32)
    x = torch.from_numpy(x).to(fmodel.device)
    y = fmodel.forward(x).argmax(axis=-1)

    attack = GaussianBlurAttack(fmodel, channel_axis=1)
    advs = attack(x, y)

    perturbations = ep.astensor(advs - x)
    norms = flatten(perturbations).square().sum(axis=-1).sqrt()
    y_advs = fmodel.forward(advs).argmax(axis=-1)

    assert x.shape == advs.shape
    assert norms.max().item() <= 20.0 + 1e-7
    assert (y_advs == y).float().mean() < 1
