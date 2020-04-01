from typing import List, Tuple
import pytest
import eagerpy as ep

import foolbox as fbn
import foolbox.attacks as fa

L2 = fbn.types.L2
Linf = fbn.types.Linf


def get_attack_id(x: fbn.Attack) -> str:
    return repr(x)


targeted_attacks: List[fbn.Attack] = [
    fa.LocalSearchAttack(t=150, p=0.5, d=5),
]


@pytest.mark.parametrize("attack", targeted_attacks, ids=get_attack_id)
def test_targeted_attacks(
    mnist_fmodel_and_data_ext: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
    attack: fbn.Attack,
) -> None:

    (fmodel, x, y), real = mnist_fmodel_and_data_ext

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    num_classes = fmodel(x).shape[-1]
    target_classes = (y + 1) % num_classes
    criterion = fbn.TargetedMisclassification(target_classes)
    adv_before_attack = criterion(x, fmodel(x))
    assert not adv_before_attack.all()

    advs, _, _ = attack(fmodel, x, criterion, epsilons=None)
    adv_after_attack = criterion(advs, fmodel(advs))
    assert adv_after_attack.sum().item() > adv_before_attack.sum().item()
