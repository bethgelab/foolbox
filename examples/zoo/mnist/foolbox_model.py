#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
from foolbox.models import PyTorchModel
from foolbox.utils import accuracy, samples


def create() -> PyTorchModel:
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.Linear(128, 10),
    )
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mnist_cnn.pth")
    model.load_state_dict(torch.load(path))  # type: ignore
    model.eval()
    preprocessing = dict(mean=0.1307, std=0.3081)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return fmodel


if __name__ == "__main__":
    # test the model
    fmodel = create()
    images, labels = samples(fmodel, dataset="mnist", batchsize=20)
    print(accuracy(fmodel, images, labels))
