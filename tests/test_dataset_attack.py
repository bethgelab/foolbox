from typing import Tuple
import eagerpy as ep

import foolbox as fbn


def test_dataset_attack(
    fmodel_and_data_ext_for_attacks: Tuple[
        Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool
    ],
) -> None:

    (fmodel, x, y), _ = fmodel_and_data_ext_for_attacks
    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    attack = fbn.attacks.DatasetAttack()
    attack.feed(fmodel, x)

    assert fbn.accuracy(fmodel, x, y) > 0

    advs, _, success = attack(fmodel, x, y, epsilons=None)
    assert success.shape == (len(x),)
    assert success.all()
    assert fbn.accuracy(fmodel, advs, y) == 0
