import numpy as np
import eagerpy as ep
from inspect import signature

from .models import Model
from .devutils import flatten


def evaluate_l2(fmodel: Model, inputs, labels, *, attacks, epsilons):
    x, y = ep.astensors(inputs, labels)
    del inputs, labels

    attack_success = np.zeros((len(attacks), len(epsilons), len(x)), dtype=np.float32)

    for i, attack in enumerate(attacks):
        # TODO: update to new attack interface
        attack = attack(fmodel)
        sig = signature(attack.__call__)
        minimizing = "epsilon" not in sig.parameters

        if minimizing:
            # TODO: support hyperparameters
            xp = attack(x, y)
            logits = fmodel(xp)
            predictions = logits.argmax(axis=-1)
            correct = (predictions == y).float32().numpy().astype(np.bool)
            perturbations = xp - x
            norms = flatten(perturbations).square().sum(axis=-1).sqrt().numpy()
            for j, epsilon in enumerate(epsilons):
                attack_success[i, j] = np.logical_and(
                    np.logical_not(correct), norms <= epsilon
                )
        else:
            for j, epsilon in enumerate(epsilons):
                xp = attack(x, y, epsilon=epsilon)
                logits = fmodel(xp)
                predictions = logits.argmax(axis=-1)
                correct = (predictions == y).float32().numpy().astype(np.bool)
                perturbations = xp - x
                norms = flatten(perturbations).square().sum(axis=-1).sqrt().numpy()
                # TODO: relax this norm check or pass a slightly stricter norm to the attack
                attack_success[i, j] = np.logical_and(
                    np.logical_not(correct), norms <= epsilon
                ).astype(np.float32)

    robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
    return attack_success, robust_accuracy
