from typing import List, Tuple, Optional
import pytest
import eagerpy as ep
import numpy as np

import foolbox as fbn
import foolbox.attacks as fa
from foolbox.attacks.spatial_attack_transformations import rotate_and_shift

def get_attack_id(x: Tuple[fbn.Attack, bool, bool]) -> str:
    return repr(x[0])

# attack
attacks: List[Tuple[fbn.Attack, bool]] = [
    (fa.SpatialAttack(), False),
    (fa.SpatialAttack(grid_search=False), False)
]


@pytest.mark.parametrize("attack_grad_real", attacks, ids=get_attack_id)
def test_spatial_attacks(
    fmodel_and_data_ext_for_attacks: Tuple[Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool],
    attack_grad_real: Tuple[fbn.Attack, Optional[float], bool],
) -> None:

    attack, attack_uses_grad = attack_grad_real
    (fmodel, x, y), real = fmodel_and_data_ext_for_attacks
    if  not real:
        pytest.skip()

    if isinstance(x, ep.NumPyTensor) and attack_uses_grad:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    acc = fbn.accuracy(fmodel, x, y)
    assert acc > 0

    advs, _, _ = attack(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) < acc
