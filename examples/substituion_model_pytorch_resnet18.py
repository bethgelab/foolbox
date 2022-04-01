#!/usr/bin/env python3
# mypy: no-disallow-untyped-defs
"""
Sometimes one wants to replace the gradient of a model with a different gradient
from another model to make the attack more reliable. That is, the forward pass
should go through model 1, but the backward pass should go through model 2.
This example shows how that can be done in Foolbox.
"""
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
from foolbox.attacks.base import get_criterion


def main() -> None:
    # instantiate a model (could also be a TensorFlow or JAX model)
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # replace the gradient with the gradient from another model
    model2 = fmodel  # demo, we just use the same model

    # TODO: this is still a bit annoying because we need
    # to overwrite run to get the labels
    class Attack(LinfPGD):
        def value_and_grad(self, loss_fn, x):
            val1 = loss_fn(x)
            loss_fn2 = self.get_loss_fn(model2, self.labels)
            _, grad2 = ep.value_and_grad(loss_fn2, x)
            return val1, grad2

        def run(self, model, inputs, criterion, *, epsilon, **kwargs):
            criterion_ = get_criterion(criterion)
            self.labels = criterion_.labels
            return super().run(model, inputs, criterion_, epsilon=epsilon, **kwargs)

    # apply the attack
    attack = Attack()
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

    # we can also manually check this
    # we will use the clipped advs instead of the raw advs, otherwise
    # we would need to check if the perturbation sizes are actually
    # within the specified epsilon bound
    print()
    print("we can also manually check this:")
    print()
    print("robust accuracy for perturbations with")
    for eps, advs_ in zip(epsilons, clipped_advs):
        acc2 = accuracy(fmodel, advs_, labels)
        print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
        print("    perturbation sizes:")
        perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break


if __name__ == "__main__":
    main()
