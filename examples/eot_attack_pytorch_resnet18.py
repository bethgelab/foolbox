#!/usr/bin/env python3
"""
A simple example that demonstrates how to run Expectation over Transformation
coupled with any attack, on a Resnet-18 PyTorch model.
"""
import torch
from torch import Tensor
import torchvision.models as models
import torchvision.transforms as transforms
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
from foolbox.models import ExpectationOverTransformationWrapper


class RandomizedResNet18(torch.nn.Module):
    def __init__(self) -> None:

        super().__init__()

        # base model
        self.model = models.resnet18(pretrained=True)

        # random apply rotation
        self.transforms = transforms.RandomRotation(degrees=25)

    def forward(self, x: Tensor) -> Tensor:

        # random transform
        x = self.transforms(x)

        return self.model(x)


def main() -> None:
    # instantiate a model (could also be a TensorFlow or JAX model)
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))

    print("Testing attack on the base model (no transformations applied)")
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # apply an attack with different eps
    attack = LinfPGD()
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.02,
        0.03,
        0.1,
        0.3,
        0.5,
        1.0,
    ]

    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    robust_accuracy = 1 - success.float32().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    # Let's apply the same LinfPGD attack, but on a model with random transformations
    rand_model = RandomizedResNet18().eval()
    fmodel = PyTorchModel(rand_model, bounds=(0, 1), preprocessing=preprocessing)
    seed = 1111

    print("#" * 40)
    print("Testing attack on the randomized model (random rotation applied)")

    # Note: accuracy may slightly decrease, depending on seed
    torch.manual_seed(seed)
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # test the base attack on the randomized model
    print("robust accuracy for perturbations with")
    for eps in epsilons:

        # reset seed to have the same perturbations in each attack
        torch.manual_seed(seed)
        _, _, success = attack(fmodel, images, labels, epsilons=eps)

        # calculate and report the robust accuracy
        # the attack is performing worse on the randomized models, since gradient computation is affected!
        robust_accuracy = 1 - success.float32().mean(axis=-1)
        print(f"  Linf norm ≤ {eps:<6}: {robust_accuracy.item() * 100:4.1f} %")

    # Now, Let's use Expectation Over Transformation to counter the randomization
    eot_model = ExpectationOverTransformationWrapper(fmodel, n_steps=16)

    print("#" * 40)
    print("Testing EoT attack on the randomized model (random crop applied)")
    torch.manual_seed(seed)
    clean_acc = accuracy(eot_model, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    print("robust accuracy for perturbations with")
    for eps in epsilons:
        # reset seed to have the same perturbations in each attack
        torch.manual_seed(seed)
        _, _, success = attack(eot_model, images, labels, epsilons=eps)

        # calculate and report the robust accuracy
        # with EoT, the base attack is working again!
        robust_accuracy = 1 - success.float32().mean(axis=-1)
        print(f"  Linf norm ≤ {eps:<6}: {robust_accuracy.item() * 100:4.1f} %")


if __name__ == "__main__":
    main()
