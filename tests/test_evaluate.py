import numpy as np
import torch
import torch.nn as nn

from foolbox.ext.native import evaluate_l2
from foolbox.ext.native.utils import accuracy

from foolbox.ext.native.models import PyTorchModel
from foolbox.ext.native.attacks import L2BasicIterativeAttack
from foolbox.ext.native.attacks import L2CarliniWagnerAttack
from foolbox.ext.native.attacks import L2ContrastReductionAttack
from foolbox.ext.native.attacks import BinarySearchContrastReductionAttack
from foolbox.ext.native.attacks import LinearSearchContrastReductionAttack


def test_evaluate():
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

    attacks = [
        L2BasicIterativeAttack,
        L2CarliniWagnerAttack,
        L2ContrastReductionAttack,
        BinarySearchContrastReductionAttack,
        LinearSearchContrastReductionAttack,
    ]
    epsilons = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

    acc = accuracy(fmodel, x, y)
    assert acc > 0
    _, robust_accuracy = evaluate_l2(fmodel, x, y, attacks=attacks, epsilons=epsilons)

    assert robust_accuracy[0] == acc
    assert robust_accuracy[-1] == 0.0
