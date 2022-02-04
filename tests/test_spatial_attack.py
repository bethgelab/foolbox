from typing import List, Tuple
import pytest
import foolbox as fbn
import foolbox.attacks as fa

from conftest import ModeAndDataAndDescription


def get_attack_id(x: fbn.Attack) -> str:
    return repr(x)


# attack
attacks: List[Tuple[fbn.Attack, bool]] = [
    (fa.SpatialAttack(), False),
    (fa.SpatialAttack(grid_search=False), False),
    (fa.SpatialAttack(grid_search=False), True),
]


@pytest.mark.parametrize("attack_grad_real", attacks, ids=get_attack_id)
def test_spatial_attacks(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_grad_real: Tuple[fbn.Attack, bool],
) -> None:

    attack, repeated = attack_grad_real
    if repeated:
        attack = attack.repeat(2)
    (fmodel, x, y), real, _ = fmodel_and_data_ext_for_attacks
    if not real:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    acc = fbn.accuracy(fmodel, x, y)
    assert acc > 0
    advs, _, _ = attack(fmodel, x, y)  # type: ignore
    assert fbn.accuracy(fmodel, advs, y) < acc
