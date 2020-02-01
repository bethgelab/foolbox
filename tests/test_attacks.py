from typing import List, Tuple
import pytest
import eagerpy as ep

import foolbox.ext.native as fbn

L2 = fbn.types.L2

attacks: List[fbn.Attack] = [
    fbn.attacks.InversionAttack(),
    fbn.attacks.L2ContrastReductionAttack(L2(100.0)),
    fbn.attacks.BinarySearchContrastReductionAttack(),
    fbn.attacks.LinearSearchContrastReductionAttack(),
]


@pytest.mark.parametrize("attack", attacks)
def test_init_and_call(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor], attack: fbn.Attack
) -> None:
    fmodel, x, y = fmodel_and_data
    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    advs = attack(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
