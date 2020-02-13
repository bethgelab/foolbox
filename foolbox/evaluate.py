from typing import Tuple, Any, Sequence
import numpy as np
import eagerpy as ep

from .models import Model
from .attacks import Attack


def evaluate(
    fmodel: Model,
    inputs: Any,
    labels: Any,
    *,
    attacks: Sequence[Attack],
    epsilons: Sequence[float],
) -> Tuple[Any, Any]:
    x, y = ep.astensors(inputs, labels)
    del inputs, labels

    attack_success = np.zeros((len(attacks), len(epsilons), len(x)), dtype=np.bool)
    for i, attack in enumerate(attacks):
        _, success = attack(fmodel, x, y, epsilons=epsilons)
        assert success.shape == (len(epsilons), len(x))
        success_ = success.numpy()
        assert success_.dtype == np.bool
        attack_success[i] = success_

    robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
    return attack_success, robust_accuracy
