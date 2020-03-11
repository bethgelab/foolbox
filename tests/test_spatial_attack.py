from typing import List, Tuple
import pytest
import eagerpy as ep
import foolbox as fbn
import foolbox.attacks as fa


def get_attack_id(x: fbn.Attack) -> str:
    return repr(x)


# attack
attacks: List[fbn.Attack] = [
    fa.SpatialAttack(),
    fa.SpatialAttack(grid_search=False),
]


@pytest.mark.parametrize("attack_grad_real", attacks, ids=get_attack_id)
def test_spatial_attacks(
    fmodel_and_data_ext_for_attacks: Tuple[
        Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool
    ],
    attack_grad_real: fbn.Attack,
) -> None:

    attack = attack_grad_real
    (fmodel, x, y), real = fmodel_and_data_ext_for_attacks
    if not real:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    acc = fbn.accuracy(fmodel, x, y)
    assert acc > 0
    advs, _, _ = attack(fmodel, x, y)  # type: ignore
    assert fbn.accuracy(fmodel, advs, y) < acc
