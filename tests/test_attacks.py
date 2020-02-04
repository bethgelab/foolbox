from typing import List, Tuple
import pytest
import eagerpy as ep

import foolbox.ext.native as fbn

L2 = fbn.types.L2

attacks: List[fbn.Attack] = [
    fbn.attacks.InversionAttack(),
    fbn.attacks.L2ContrastReductionAttack(L2(100.0)),
    fbn.attacks.BinarySearchContrastReductionAttack(),
    fbn.attacks.LinearSearchContrastReductionAttack(steps=20),
    fbn.attacks.L2CarliniWagnerAttack(binary_search_steps=3, steps=50),
]


@pytest.mark.parametrize("attack", attacks)
def test_untargeted_attacks(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor], attack: fbn.Attack
) -> None:

    fmodel, x, y = fmodel_and_data
    if isinstance(x, ep.NumPyTensor) and isinstance(
        attack, fbn.attacks.L2CarliniWagnerAttack
    ):
        pytest.skip()
    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    advs = attack(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)


targeted_attacks: List[fbn.Attack] = [
    fbn.attacks.L2CarliniWagnerAttack(binary_search_steps=3, steps=50)
]


@pytest.mark.parametrize("attack", targeted_attacks)
def test_targeted_attacks(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor], attack: fbn.Attack
) -> None:

    fmodel, x, y = fmodel_and_data
    if isinstance(x, ep.NumPyTensor) and isinstance(
        attack, fbn.attacks.L2CarliniWagnerAttack
    ):
        pytest.skip()
    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    num_classes = fmodel(x).shape[-1]
    target_classes = (y + 1) % num_classes
    criterion = fbn.TargetedMisclassification(target_classes)
    advs = attack(fmodel, x, criterion)

    adv_before_attack = criterion(x, fmodel(x))
    adv_after_attack = criterion(advs, fmodel(advs))
    assert adv_after_attack.sum().item() > adv_before_attack.sum().item()
