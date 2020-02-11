from typing import Tuple
import eagerpy as ep

import foolbox as fbn


def test_dataset_attack(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
) -> None:

    fmodel, x, y = fmodel_and_data
    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    attack = fbn.attacks.DatasetAttack()
    attack.feed(fmodel, x)

    advs = attack(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
