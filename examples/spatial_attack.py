#!/usr/bin/env python3
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np


if __name__ == "__main__":
    # instantiate a model
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    print("accuracy")
    print(accuracy(fmodel, images, labels))
    print("")

    # stops early if shifts and translations for all images are found
    attack = fa.spatial_attack.SpatialAttack(
        max_translation=6,  # 5px so x in [x-5, x+5] and y in [y-5, y+5]
        num_translations=6,  # number of translations in x, y.
        max_rotation=20,  # +- rotation in degrees
        num_rotations=5,  # number of rotations
        # max total iterations = num_rotations * num_translations**2
    )

    xp_, _, success = attack(fmodel, images, labels)
    print(
        "attack success in specified rotation in translation bounds",
        success.numpy().astype(np.float32).mean() * 100,
        " %",
    )
