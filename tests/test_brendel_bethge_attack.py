from typing import Tuple, Union, List, Any
import eagerpy as ep

import foolbox as fbn
import foolbox.attacks as fa
from foolbox.devutils import flatten
from foolbox.attacks.brendel_bethge import BrendelBethgeAttack
import pytest

from conftest import ModeAndDataAndDescription


def get_attack_id(x: Tuple[BrendelBethgeAttack, Union[int, float]]) -> str:
    return repr(x[0])


attacks: List[Tuple[fa.Attack, Union[int, float]]] = [
    (fa.L0BrendelBethgeAttack(steps=20), 0),
    (fa.L1BrendelBethgeAttack(steps=20), 1),
    (fa.L2BrendelBethgeAttack(steps=20), 2),
    (fa.LinfinityBrendelBethgeAttack(steps=20), ep.inf),
]


@pytest.mark.parametrize("attack_and_p", attacks, ids=get_attack_id)
def test_brendel_bethge_untargeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_and_p: Tuple[BrendelBethgeAttack, Union[int, float]],
) -> None:
    if request.config.option.skipslow:
        pytest.skip()

    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    if low_dimensional_input:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    init_attack = fa.DatasetAttack()
    init_attack.feed(fmodel, x)
    init_advs = init_attack.run(fmodel, x, y)

    attack, p = attack_and_p
    advs = attack.run(fmodel, x, y, starting_points=init_advs)

    init_norms = ep.norms.lp(flatten(init_advs - x), p=p, axis=-1)
    norms = ep.norms.lp(flatten(advs - x), p=p, axis=-1)

    is_smaller = norms < init_norms

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller.any()
