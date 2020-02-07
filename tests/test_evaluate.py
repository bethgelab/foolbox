from typing import Tuple
import pytest
import foolbox as fbn
import eagerpy as ep


def test_evaluate(fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]) -> None:
    pytest.skip()
    assert False
    fmodel, x, y = fmodel_and_data  # type: ignore

    attacks = [
        # L2BasicIterativeAttack,
        # L2CarliniWagnerAttack,
        # L2ContrastReductionAttack,
        # BinarySearchContrastReductionAttack,
        # LinearSearchContrastReductionAttack,
    ]
    epsilons = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

    acc = fbn.accuracy(fmodel, x, y)
    assert acc > 0

    _, robust_accuracy = fbn.evaluate_l2(
        fmodel, x, y, attacks=attacks, epsilons=epsilons
    )
    assert robust_accuracy[0] == acc
    assert robust_accuracy[-1] == 0.0
