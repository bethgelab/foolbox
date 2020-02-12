from typing import List, Tuple
import pytest
import eagerpy as ep

import foolbox as fbn
import foolbox.attacks as fa

L2 = fbn.types.L2
Linf = fbn.types.Linf


def get_attack_id(x: Tuple[fbn.Attack, bool, bool]) -> str:
    return repr(x[0])


# attack, attack_uses_grad, requires_real_model
attacks: List[Tuple[fbn.Attack, bool, bool]] = [
    (fa.DDNAttack(), True, False),
    (fa.InversionAttack(), False, False),
    (fa.L2ContrastReductionAttack(L2(100.0)), False, False),
    (fa.BinarySearchContrastReductionAttack(binary_search_steps=15), False, False),
    (fa.LinearSearchContrastReductionAttack(steps=20), False, False),
    (fa.L2CarliniWagnerAttack(binary_search_steps=3, steps=20), True, False),
    (fa.EADAttack(binary_search_steps=3, steps=20), True, False),
    (fa.EADAttack(binary_search_steps=3, steps=20, decision_rule="L1"), True, False),
    (fa.NewtonFoolAttack(steps=20), True, False),
    (fa.L2ContrastReductionAttack(L2(100.0)).repeat(3), False, False),
    (fa.VirtualAdversarialAttack(iterations=50, xi=1, epsilon=10), True, False),
    (fa.L2BasicIterativeAttack(L2(100.0), stepsize=5.0, steps=10), True, False),
    (fa.LinfBasicIterativeAttack(Linf(1.0), stepsize=5.0, steps=10), True, False),
    (fa.PGD(Linf(1.0), stepsize=5.0, steps=10), True, False,),
    (fa.L2FastGradientAttack(L2(100.0)), True, False),
    (fa.LinfFastGradientAttack(Linf(100.0)), True, False),
    (fa.GaussianBlurAttack(steps=10), True, True),
    (fa.L2DeepFoolAttack(steps=50, loss="logits"), True, False),
    (fa.L2DeepFoolAttack(steps=50, loss="crossentropy"), True, False),
    (fa.LinfDeepFoolAttack(steps=50), True, False),
    (fa.SaltAndPepperNoiseAttack(steps=50), True, False),
    (fa.BoundaryAttack(steps=50), False, False),
    (fa.LinearSearchBlendedUniformNoiseAttack(steps=50), False, False),
    (fa.L2AdditiveGaussianNoiseAttack(epsilon=1500.0), False, False),
    (fa.LinfAdditiveUniformNoiseAttack(epsilon=10.0), False, False),
    (fa.L2RepeatedAdditiveGaussianNoiseAttack(epsilon=1000.0), False, False),
    (fa.L2RepeatedAdditiveUniformNoiseAttack(epsilon=1000.0), False, False),
    (fa.LinfRepeatedAdditiveUniformNoiseAttack(epsilon=3.0), False, False),
]


@pytest.mark.parametrize("attack_and_grad", attacks, ids=get_attack_id)
def test_untargeted_attacks(
    fmodel_and_data_ext_for_attacks: Tuple[
        Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool
    ],
    attack_and_grad: Tuple[fbn.Attack, bool, bool],
) -> None:

    attack, attack_uses_grad, requires_real_model = attack_and_grad
    (fmodel, x, y), real = fmodel_and_data_ext_for_attacks
    if requires_real_model and not real:
        pytest.skip()

    if isinstance(x, ep.NumPyTensor) and attack_uses_grad:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    advs = attack(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)


targeted_attacks: List[Tuple[fbn.Attack, bool, bool]] = [
    (fa.L2CarliniWagnerAttack(binary_search_steps=3, steps=20), True, False),
    (fa.DDNAttack(), True, False),
    # TODO: targeted EADAttack currently fails repeatedly on MobileNetv2
    # (
    #     fa.EADAttack(
    #         binary_search_steps=3, steps=20, abort_early=True, regularization=0
    #     ),
    #     True,
    #     False,
    # ),
]


@pytest.mark.parametrize("attack_and_grad", targeted_attacks, ids=get_attack_id)
def test_targeted_attacks(
    fmodel_and_data_ext_for_attacks: Tuple[
        Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool
    ],
    attack_and_grad: Tuple[fbn.Attack, bool, bool],
) -> None:

    attack, attack_uses_grad, requires_real_model = attack_and_grad
    (fmodel, x, y), real = fmodel_and_data_ext_for_attacks
    if requires_real_model and not real:
        pytest.skip()

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
