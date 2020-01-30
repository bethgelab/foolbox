import eagerpy as ep
import numpy as np
from inspect import signature

from .devutils import flatten


def evaluate_l2(fmodel, inputs, labels, *, attacks, epsilons):
    x = ep.astensor(inputs)
    y = ep.astensor(labels)

    attack_success = np.zeros((len(attacks), len(epsilons), len(x)), dtype=np.float32)

    for i, attack in enumerate(attacks):
        attack = attack(fmodel)
        sig = signature(attack.__call__)
        minimizing = "epsilon" not in sig.parameters

        if minimizing:
            # TODO: support hyperparameters
            xp = ep.astensor(attack(x.tensor, y.tensor))
            logits = fmodel(xp)
            predictions = logits.argmax(axis=-1)
            correct = (predictions == labels).float32().numpy().astype(np.bool)
            perturbations = xp - x
            norms = flatten(perturbations).square().sum(axis=-1).sqrt().numpy()
            for j, epsilon in enumerate(epsilons):
                attack_success[i, j] = np.logical_and(
                    np.logical_not(correct), norms <= epsilon
                )
        else:
            for j, epsilon in enumerate(epsilons):
                xp = ep.astensor(attack(x.tensor, y.tensor, epsilon=epsilon))
                logits = fmodel(xp.tensor)
                predictions = logits.argmax(axis=-1)
                correct = (predictions == labels).float32().numpy().astype(np.bool)
                perturbations = xp - x
                norms = flatten(perturbations).square().sum(axis=-1).sqrt().numpy()
                # TODO: relax this norm check or pass a slightly stricter norm to the attack
                attack_success[i, j] = np.logical_and(
                    np.logical_not(correct), norms <= epsilon
                ).astype(np.float32)

    robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
    return attack_success, robust_accuracy
