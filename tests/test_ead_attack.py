import eagerpy as ep
import numpy as np
import torch
import torch.nn as nn

from foolbox.ext.native.devutils import flatten
from foolbox.ext.native.models import PyTorchModel
from foolbox.ext.native.attacks import EADAttack
from foolbox.ext.native.attacks import L2CarliniWagnerAttack


def test_ead_attack_cw():
    channels = 3
    batch_size = 8
    h = w = 32
    bounds = (0, 1)

    class Model(nn.Module):
        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            return x

    model = Model().eval()
    fmodel = PyTorchModel(model, bounds=bounds)

    np.random.seed(0)
    x = np.random.uniform(*bounds, size=(batch_size, channels, h, w)).astype(np.float32)
    x = torch.from_numpy(x).to(fmodel.device)
    y = fmodel.forward(x).argmax(axis=-1)

    attack = EADAttack(fmodel)
    cw_attack = L2CarliniWagnerAttack(fmodel)
    advs = attack(x, y, regularization=0, binary_search_steps=5, max_iterations=1000)
    advs_cw = cw_attack(x, y, binary_search_steps=5, max_iterations=1000)

    perturbations = ep.astensor(advs - x)
    perturbations_cw = ep.astensor(advs_cw - x)
    y_advs = fmodel.forward(advs).argmax(axis=-1)
    y_advs_cw = fmodel.forward(advs).argmax(axis=-1)

    diff = flatten(perturbations - perturbations_cw).square().sum(axis=-1).sqrt()

    assert x.shape == advs.shape
    assert diff.max().item() <= 40.0 + 1e-7
    assert (y_advs == y_advs_cw).float().mean() == 1


def test_ead_attack():
    channels = 3
    batch_size = 8
    h = w = 32
    bounds = (0, 1)

    class Model(nn.Module):
        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            return x

    model = Model().eval()
    fmodel = PyTorchModel(model, bounds=bounds)

    np.random.seed(0)
    x = np.random.uniform(*bounds, size=(batch_size, channels, h, w)).astype(np.float32)
    x = torch.from_numpy(x).to(fmodel.device)
    y = fmodel.forward(x).argmax(axis=-1)

    attack = EADAttack(fmodel)
    advs = attack(x, y, binary_search_steps=5, max_iterations=1000)

    perturbations = ep.astensor(advs - x)
    norms = flatten(perturbations).square().sum(axis=-1).sqrt()
    y_advs = fmodel.forward(advs).argmax(axis=-1)

    assert x.shape == advs.shape
    assert norms.max().item() <= 40.0 + 1e-7
    assert (y_advs == y).float().mean() < 1
