from typing import List, Any
import eagerpy as ep

import foolbox as fbn
import foolbox.attacks as fa
from foolbox.devutils import flatten
import pytest

from conftest import ModeAndDataAndDescription


def get_attack_id(x: fa.Attack) -> str:
    return repr(x)


attacks: List[fa.Attack] = [
    fa.PointwiseAttack(),
    fa.PointwiseAttack(l2_binary_search=False),
]


@pytest.mark.parametrize("attack", attacks, ids=get_attack_id)
def test_pointwise_untargeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack: fa.PointwiseAttack,
) -> None:
    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if not low_dimensional_input or not real:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    init_attack = fa.SaltAndPepperNoiseAttack(steps=50)
    init_advs = init_attack.run(fmodel, x, y)

    advs = attack.run(fmodel, x, y, starting_points=init_advs)

    init_norms_l0 = ep.norms.lp(flatten(init_advs - x), p=0, axis=-1)
    norms_l0 = ep.norms.lp(flatten(advs - x), p=0, axis=-1)

    init_norms_l2 = ep.norms.lp(flatten(init_advs - x), p=2, axis=-1)
    norms_l2 = ep.norms.lp(flatten(advs - x), p=2, axis=-1)

    is_smaller_l0 = norms_l0 < init_norms_l0
    is_smaller_l2 = norms_l2 < init_norms_l2

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert is_smaller_l2.any()
    assert is_smaller_l0.any()


@pytest.mark.parametrize("attack", attacks, ids=get_attack_id)
def test_pointwise_targeted_attack(
    request: Any,
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack: fa.PointwiseAttack,
) -> None:
    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if not low_dimensional_input or not real:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    init_attack = fa.SaltAndPepperNoiseAttack(steps=50)
    init_advs = init_attack.run(fmodel, x, y)

    logits = fmodel(init_advs)
    num_classes = logits.shape[-1]
    target_classes = logits.argmax(-1)
    target_classes = ep.where(
        target_classes == y, (target_classes + 1) % num_classes, target_classes
    )
    criterion = fbn.TargetedMisclassification(target_classes)

    advs = attack.run(fmodel, x, criterion, starting_points=init_advs)

    init_norms_l0 = ep.norms.lp(flatten(init_advs - x), p=0, axis=-1)
    norms_l0 = ep.norms.lp(flatten(advs - x), p=0, axis=-1)

    init_norms_l2 = ep.norms.lp(flatten(init_advs - x), p=2, axis=-1)
    norms_l2 = ep.norms.lp(flatten(advs - x), p=2, axis=-1)

    is_smaller_l0 = norms_l0 < init_norms_l0
    is_smaller_l2 = norms_l2 < init_norms_l2

    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) <= fbn.accuracy(fmodel, init_advs, y)
    assert fbn.accuracy(fmodel, advs, target_classes) > fbn.accuracy(
        fmodel, x, target_classes
    )
    assert fbn.accuracy(fmodel, advs, target_classes) >= fbn.accuracy(
        fmodel, init_advs, target_classes
    )
    assert is_smaller_l2.any()
    assert is_smaller_l0.any()
