#!/usr/bin/env python3
# mypy: no-disallow-untyped-defs
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
from foolbox.attacks.base import get_criterion


if __name__ == "__main__":
    # instantiate a model
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    print(accuracy(fmodel, images, labels))

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
    epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
    advs, _, success = attack(fmodel, images, labels, epsilons=epsilons)

    # calculate and report the robust accuracy
    robust_accuracy = 1 - success.float32().mean(axis=-1)
    for eps, acc in zip(epsilons, robust_accuracy):
        print(eps, acc.item())

    # we can also manually check this
    for eps, advs_ in zip(epsilons, advs):
        print(eps, accuracy(fmodel, advs_, labels))
        # but then we also need to look at the perturbation sizes
        # and check if they are smaller than eps
        print((advs_ - images).norms.linf(axis=(1, 2, 3)).numpy())
