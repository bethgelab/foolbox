import eagerpy as ep
import numpy as np
import torch
import torch.nn as nn

from foolbox.ext.native.devutils import flatten
from foolbox.ext.native.models import PyTorchModel
from foolbox.ext.native.attacks import (
    L0BrendelBethgeAttack,
    L1BrendelBethgeAttack,
    L2BrendelBethgeAttack,
    LinfinityBrendelBethgeAttack,
)


def test_l0_brendel_bethge_attack():
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

    attack = L0BrendelBethgeAttack(fmodel)
    advs = attack(x, y, steps=250, lr_num_decay=10)

    perturbations = ep.astensor(advs - x)
    norms = (flatten(perturbations).square() > 1e-5).sum(axis=-1)
    y_advs = fmodel.forward(advs).argmax(axis=-1)

    assert x.shape == advs.shape
    assert norms.max().item() <= 32 * 32 * 3 / 2
    assert (y_advs == y).float().mean() < 1e-5


def test_l1_brendel_bethge_attack():
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

    attack = L1BrendelBethgeAttack(fmodel)
    advs = attack(x, y, steps=100, lr_num_decay=10)

    perturbations = ep.astensor(advs - x)
    norms = flatten(perturbations).abs().sum(axis=-1)
    y_advs = fmodel.forward(advs).argmax(axis=-1)

    assert x.shape == advs.shape
    assert norms.max().item() <= 32 * 32 * 3 / 2
    assert (y_advs == y).float().mean() < 1e-5


def test_l2_brendel_bethge_attack():
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

    attack = L2BrendelBethgeAttack(fmodel)
    advs = attack(x, y, steps=100, lr_num_decay=10)

    perturbations = ep.astensor(advs - x)
    norms = flatten(perturbations).square().sum(axis=-1).sqrt()
    y_advs = fmodel.forward(advs).argmax(axis=-1)

    assert x.shape == advs.shape
    assert norms.max().item() <= 40.0 + 1e-7
    assert (y_advs == y).float().mean() < 1e-5


def test_linfinity_brendel_bethge_attack():
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

    attack = LinfinityBrendelBethgeAttack(fmodel)
    advs = attack(x, y, steps=100, lr_num_decay=10)

    perturbations = ep.astensor(advs - x)
    norms = flatten(perturbations).abs().max(axis=-1)
    y_advs = fmodel.forward(advs).argmax(axis=-1)

    assert x.shape == advs.shape
    assert norms.max().item() <= 0.5 + 1e-5
    assert (y_advs == y).float().mean() < 1
