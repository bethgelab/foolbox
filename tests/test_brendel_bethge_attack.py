from typing import Tuple, Union, List
import eagerpy as ep

import foolbox as fbn
import foolbox.attacks as fa
from foolbox.devutils import flatten
import pytest


def get_attack_id(x: Tuple[fa.Attack, Union[int, float]]) -> str:
    return repr(x[0])


attacks: List[Tuple[fa.Attack, Union[int, float]]] = [
    (fa.L0BrendelBethgeAttack(steps=50), 0),
    (fa.L1BrendelBethgeAttack(steps=50), 1),
    (fa.L2BrendelBethgeAttack(steps=50), 2),
    (fa.LinfinityBrendelBethgeAttack(steps=50), ep.inf),
]


@pytest.mark.parametrize("attack_and_p", attacks, ids=get_attack_id)
def test_brendel_bethge_untargeted_attack(
    fmodel_and_data_ext_for_attacks: Tuple[
        Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool
    ],
    attack_and_p: Tuple[fbn.Attack, Union[int, float]],
) -> None:
    (fmodel, x, y), real = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    init_attack = fa.LinearSearchBlendedUniformNoiseAttack(directions=100, steps=10)
    init_advs = init_attack.run(fmodel, x, y)

    attack, p = attack_and_p
    advs = attack.run(fmodel, x, y, starting_points=init_advs)

    mean_l2_init = ep.norms.lp(flatten(init_advs - x), p=p, axis=-1).mean()
    mean_l2 = ep.norms.lp(flatten(advs - x), p=p, axis=-1).mean()

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert mean_l2 < mean_l2_init
