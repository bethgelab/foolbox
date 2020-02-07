from typing import List, Tuple
import pytest
import eagerpy as ep

import foolbox.ext.native as fbn

L2 = fbn.types.L2

attacks: List[Tuple[fbn.Attack, bool]] = [
    (fbn.attacks.DDNAttack(), True),
    (fbn.attacks.DDNAttack(rescale=True), True),
    (fbn.attacks.InversionAttack(), False),
    (fbn.attacks.L2ContrastReductionAttack(L2(100.0)), False),
    (fbn.attacks.BinarySearchContrastReductionAttack(binary_search_steps=15), False),
    (fbn.attacks.LinearSearchContrastReductionAttack(steps=20), False),
    (fbn.attacks.L2CarliniWagnerAttack(binary_search_steps=3, steps=20), True),
    (fbn.attacks.EADAttack(binary_search_steps=3, steps=20), True),
    (fbn.attacks.EADAttack(binary_search_steps=3, steps=20, decision_rule="L1"), True),
    (fbn.attacks.NewtonFoolAttack(steps=20), True),
    (fbn.attacks.L2ContrastReductionAttack(L2(100.0)).repeat(3), False),
]


@pytest.mark.parametrize("attack_and_grad", attacks)
def test_untargeted_attacks(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
    attack_and_grad: Tuple[fbn.Attack, bool],
) -> None:

    attack, attack_uses_grad = attack_and_grad
    fmodel, x, y = fmodel_and_data

    if isinstance(x, ep.NumPyTensor) and attack_uses_grad:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    advs = attack(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)


targeted_attacks: List[Tuple[fbn.Attack, bool]] = [
    (fbn.attacks.L2CarliniWagnerAttack(binary_search_steps=3, steps=20), True),
    (fbn.attacks.DDNAttack(), True),
    (fbn.attacks.EADAttack(binary_search_steps=3, steps=20), True),
]


@pytest.mark.parametrize("attack_and_grad", targeted_attacks)
def test_targeted_attacks(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
    attack_and_grad: Tuple[fbn.Attack, bool],
) -> None:

    attack, attack_uses_grad = attack_and_grad
    fmodel, x, y = fmodel_and_data

    if isinstance(x, ep.NumPyTensor) and attack_uses_grad:
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
