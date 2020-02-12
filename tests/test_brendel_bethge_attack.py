from typing import Tuple
import eagerpy as ep

import foolbox as fbn
import foolbox.attacks as fa
from foolbox.devutils import flatten
import pytest
from foolbox.criteria import TargetedMisclassification


def test_brendel_bethge_untargeted_attack(
    fmodel_and_data_ext_for_attacks: Tuple[Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool]
) -> None:
    (fmodel, x, y), real = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    init_attack = fa.LinearSearchBlendedUniformNoiseAttack(directions=100, steps=10)
    init_advs = init_attack(fmodel, x, y)

    attack = fa.L2BrendelBethgeAttack(steps=50)
    advs = attack(fmodel, x, y, starting_points=init_advs)

    mean_l2_init = ep.norms.lp(flatten(init_advs - x), p=2, axis=-1).mean()
    mean_l2 = ep.norms.lp(flatten(advs - x), p=2, axis=-1).mean()

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert mean_l2 < mean_l2_init


def test_brendel_bethge_targeted_attack(
    fmodel_and_data_ext_for_attacks: Tuple[Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool]
) -> None:
    (fmodel, x, y), real = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    n_classes = fmodel(x).shape[-1]
    t = (y + 1) % n_classes

    init_attack = fa.LinearSearchBlendedUniformNoiseAttack(directions=100, steps=10)
    init_advs = init_attack(fmodel, x, TargetedMisclassification(t))

    attack = fa.L2BrendelBethgeAttack(steps=50)
    advs = attack(fmodel, x, TargetedMisclassification(t), starting_points=init_advs)

    mean_l2_init = ep.norms.lp(flatten(init_advs - x), p=2, axis=-1).mean()
    mean_l2 = ep.norms.lp(flatten(advs - x), p=2, axis=-1).mean()

    assert fbn.accuracy(fmodel, advs, t) > fbn.accuracy(fmodel, x, t)
    assert fbn.accuracy(fmodel, advs, t) >= fbn.accuracy(fmodel, init_advs, t)
    assert mean_l2 < mean_l2_init
