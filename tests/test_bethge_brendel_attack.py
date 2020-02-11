from typing import Tuple
import eagerpy as ep

import foolbox as fbn
import foolbox.attacks as fa
from foolbox.devutils import flatten
import pytest
from foolbox.criteria import TargetedMisclassification


def test_dataset_attack(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
) -> None:

    fmodel, x, y = fmodel_and_data

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    n_classes = fmodel(x).shape[-1]
    t = (y + 1) % n_classes

    init_attack = fa.LinearSearchBlendedUniformNoiseAttack(directions=10, steps=10)
    init_advs = init_attack(fmodel, x, TargetedMisclassification(t))

    attack = fa.L2BrendelBethgeAttack(steps=50)
    advs = attack(fmodel, x, TargetedMisclassification(t), starting_points=init_advs)

    mean_l2_init = ep.norms.lp(flatten(init_advs - x), p=2, axis=-1).mean()
    mean_l2 = ep.norms.lp(flatten(advs - x), p=2, axis=-1).mean()

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert mean_l2 < mean_l2_init
