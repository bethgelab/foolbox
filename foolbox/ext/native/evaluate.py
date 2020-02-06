from typing import List, Tuple, Any
from inspect import signature
import numpy as np
import eagerpy as ep

from .models import Model
from .attacks import Attack
from .types import L2
from .devutils import flatten


def evaluate_l2(
    fmodel: Model,
    inputs: Any,
    labels: Any,
    *,
    attacks: List[Attack],
    epsilons: List[L2],
) -> Tuple[Any, Any]:
    x, y = ep.astensors(inputs, labels)
    del inputs, labels

    attack_success = np.zeros((len(attacks), len(epsilons), len(x)), dtype=np.float32)

    for i, attack in enumerate(attacks):
        sig = signature(type(attack).__init__)
        minimizing = "epsilon" not in sig.parameters

        if minimizing:
            # TODO: support hyperparameters
            xp = attack(fmodel, x, y)
            predictions = fmodel(xp).argmax(axis=-1)
            correct = (predictions == y).float32().numpy().astype(np.bool)
            perturbations = xp - x
            norms = flatten(perturbations).square().sum(axis=-1).sqrt().numpy()
            for j, epsilon in enumerate(epsilons):
                attack_success[i, j] = np.logical_and(
                    np.logical_not(correct), norms <= epsilon
                )
        else:
            for j, epsilon in enumerate(epsilons):
                attack.epsilon = epsilon  # type: ignore
                xp = attack(fmodel, x, y)
                predictions = fmodel(xp).argmax(axis=-1)
                correct = (predictions == y).float32().numpy().astype(np.bool)
                perturbations = xp - x
                norms = flatten(perturbations).square().sum(axis=-1).sqrt().numpy()
                # TODO: relax this norm check or pass a slightly stricter norm to the attack
                attack_success[i, j] = np.logical_and(
                    np.logical_not(correct), norms <= epsilon
                ).astype(np.float32)

    robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
    return attack_success, robust_accuracy
