import pytest
import numpy as np
import torch
import torch.nn as nn

from foolbox.ext.native.models import PyTorchModel
from foolbox.ext.native.attacks import LinfinityBasicIterativeAttack
from foolbox.ext.native.attacks import L2BasicIterativeAttack
from foolbox.ext.native.attacks import L2CarliniWagnerAttack
from foolbox.ext.native.attacks import PGD


@pytest.fixture
def fmodel():
    bounds = (0, 1)

    class Model(nn.Module):
        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            return x

    model = Model()
    fmodel = PyTorchModel(model, bounds=bounds)
    return fmodel


@pytest.fixture
def fmodel_and_data(fmodel):
    channels = num_classes = 10
    batch_size = 8
    h = w = 32

    np.random.seed(0)
    x = np.random.uniform(*fmodel.bounds(), size=(batch_size, channels, h, w)).astype(
        np.float32
    )
    x = torch.from_numpy(x).to(fmodel.device)

    y = np.arange(batch_size) % num_classes
    y = torch.from_numpy(y).to(fmodel.device)
    return fmodel, x, y, batch_size, num_classes


def test_pytorch_model_forward(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    output = fmodel.forward(x)
    assert output.shape == (batch_size, num_classes)
    assert isinstance(output, torch.Tensor)


def test_pytorch_model_gradient(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    output = fmodel.gradient(x, y)
    assert output.shape == x.shape
    assert isinstance(output, torch.Tensor)


def test_pytorch_linf_basic_iterative_attack(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = fmodel.forward(x).argmax(axis=-1)

    attack = LinfinityBasicIterativeAttack(fmodel)
    advs = attack(x, y, rescale=False, epsilon=0.3)

    perturbation = advs - x
    y_advs = fmodel.forward(advs).argmax(axis=-1)

    assert x.shape == advs.shape
    assert perturbation.abs().max() <= 0.3 + 1e7
    assert (y_advs == y).float().mean() < 1


def test_pytorch_l2_basic_iterative_attack(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = fmodel.forward(x).argmax(axis=-1)

    attack = L2BasicIterativeAttack(fmodel)
    advs = attack(x, y, rescale=False, epsilon=0.3)

    perturbation = advs - x
    y_advs = fmodel.forward(advs).argmax(axis=-1)

    assert x.shape == advs.shape
    assert perturbation.abs().max() <= 0.3 + 1e7
    assert (y_advs == y).float().mean() < 1


def test_pytorch_l2_carlini_wagner_attack(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = fmodel.forward(x).argmax(axis=-1)

    attack = L2CarliniWagnerAttack(fmodel)
    advs = attack(x, y, max_iterations=100)
    y_advs = fmodel.forward(advs).argmax(axis=-1)

    assert x.shape == advs.shape
    assert (y_advs == y).float().mean() < 1


def test_pytorch_linf_pgd(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = fmodel.forward(x).argmax(axis=-1)

    attack = PGD(fmodel)
    advs = attack(x, y, rescale=False, epsilon=0.3)

    perturbation = advs - x
    y_advs = fmodel.forward(advs).argmax(axis=-1)

    assert x.shape == advs.shape
    assert perturbation.abs().max() <= 0.3 + 1e7
    assert (y_advs == y).float().mean() < 1
