#!/usr/bin/env python3
import tensorflow as tf
import eagerpy as ep
from foolbox import TensorFlowModel, accuracy, samples
from foolbox.attacks import LinfPGD


if __name__ == "__main__":
    # instantiate a model
    model = tf.keras.applications.ResNet50(weights="imagenet")
    pre = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    fmodel = TensorFlowModel(model, bounds=(0, 255), preprocessing=pre)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    print(accuracy(fmodel, images, labels))

    # apply the attack
    attack = LinfPGD()
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
