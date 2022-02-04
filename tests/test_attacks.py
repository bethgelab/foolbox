from typing import List, Optional, NamedTuple
import pytest
import eagerpy as ep

import foolbox as fbn
import foolbox.attacks as fa
from foolbox.gradient_estimators import es_gradient_estimator

from conftest import ModeAndDataAndDescription

L2 = fbn.types.L2
Linf = fbn.types.Linf


FGSM_GE = es_gradient_estimator(
    fa.FGSM, samples=100, sigma=0.03, bounds=(0, 1), clip=True
)


class AttackTestTarget(NamedTuple):
    attack: fbn.Attack
    epsilon: Optional[float] = None
    uses_grad: Optional[bool] = False
    requires_real_model: Optional[bool] = False
    requires_low_dimensional_input: Optional[bool] = False
    stochastic_attack: Optional[bool] = False


def get_attack_id(x: AttackTestTarget) -> str:
    return repr(x.attack)


# attack, eps / None, attack_uses_grad, requires_real_model
attacks: List[AttackTestTarget] = [
    AttackTestTarget(fa.DDNAttack(init_epsilon=2.0), uses_grad=True),
    AttackTestTarget(fa.InversionAttack()),
    AttackTestTarget(
        fa.InversionAttack(distance=fbn.distances.l2).repeat(3).repeat(2),
    ),
    AttackTestTarget(fa.L2ContrastReductionAttack(), L2(100.0)),
    AttackTestTarget(fa.L2ContrastReductionAttack().repeat(3), 100.0),
    AttackTestTarget(fa.BinarySearchContrastReductionAttack(binary_search_steps=15)),
    AttackTestTarget(fa.LinearSearchContrastReductionAttack(steps=20)),
    AttackTestTarget(
        fa.L2CarliniWagnerAttack(binary_search_steps=11, steps=5), uses_grad=True
    ),
    AttackTestTarget(
        fa.L2CarliniWagnerAttack(binary_search_steps=3, steps=20, confidence=2.0),
        uses_grad=True,
    ),
    AttackTestTarget(
        fa.EADAttack(binary_search_steps=10, steps=20, regularization=0), uses_grad=True
    ),
    AttackTestTarget(
        fa.EADAttack(
            binary_search_steps=10, steps=20, regularization=0, confidence=2.0
        ),
        uses_grad=True,
    ),
    AttackTestTarget(
        fa.EADAttack(
            binary_search_steps=3, steps=20, decision_rule="L1", regularization=0
        ),
        uses_grad=True,
    ),
    AttackTestTarget(
        fa.NewtonFoolAttack(steps=20),
        uses_grad=True,
        requires_low_dimensional_input=True,
    ),
    AttackTestTarget(
        fa.VirtualAdversarialAttack(steps=50, xi=1),
        10,
        uses_grad=True,
        requires_low_dimensional_input=True,
    ),
    AttackTestTarget(fa.PGD(), Linf(1.0), uses_grad=True),
    AttackTestTarget(fa.L2PGD(), L2(50.0), uses_grad=True),
    AttackTestTarget(fa.L1PGD(), 5000.0, uses_grad=True),
    AttackTestTarget(
        fa.LinfBasicIterativeAttack(abs_stepsize=0.2), Linf(1.0), uses_grad=True
    ),
    AttackTestTarget(fa.L2BasicIterativeAttack(), L2(50.0), uses_grad=True),
    AttackTestTarget(fa.L1BasicIterativeAttack(), 5000.0, uses_grad=True),
    AttackTestTarget(fa.AdamPGD(), Linf(1.0), uses_grad=True),
    AttackTestTarget(fa.L2AdamPGD(), L2(50.0), uses_grad=True),
    AttackTestTarget(fa.L1AdamPGD(), 5000.0, uses_grad=True),
    AttackTestTarget(
        fa.LinfAdamBasicIterativeAttack(abs_stepsize=0.2), Linf(1.0), uses_grad=True
    ),
    AttackTestTarget(fa.L2AdamBasicIterativeAttack(), L2(50.0), uses_grad=True),
    AttackTestTarget(fa.L1AdamBasicIterativeAttack(), 5000.0, uses_grad=True),
    AttackTestTarget(fa.SparseL1DescentAttack(), 5000.0, uses_grad=True),
    AttackTestTarget(fa.FGSM(), Linf(100.0), uses_grad=True),
    AttackTestTarget(FGSM_GE(), Linf(100.0)),
    AttackTestTarget(fa.FGM(), L2(100.0), uses_grad=True),
    AttackTestTarget(fa.L1FastGradientAttack(), 5000.0, uses_grad=True),
    AttackTestTarget(
        fa.GaussianBlurAttack(steps=10), uses_grad=True, requires_real_model=True
    ),
    AttackTestTarget(
        fa.GaussianBlurAttack(steps=10, max_sigma=224.0),
        uses_grad=True,
        requires_real_model=True,
    ),
    AttackTestTarget(fa.L2DeepFoolAttack(steps=50, loss="logits"), uses_grad=True),
    AttackTestTarget(
        fa.L2DeepFoolAttack(steps=50, loss="crossentropy"), uses_grad=True
    ),
    AttackTestTarget(fa.LinfDeepFoolAttack(steps=50), uses_grad=True),
    AttackTestTarget(fa.BoundaryAttack(steps=50)),
    AttackTestTarget(
        fa.BoundaryAttack(
            steps=110,
            init_attack=fa.LinearSearchBlendedUniformNoiseAttack(steps=50),
            update_stats_every_k=1,
        )
    ),
    AttackTestTarget(
        fa.SaltAndPepperNoiseAttack(steps=50),
        None,
        uses_grad=True,
        stochastic_attack=True,
    ),
    AttackTestTarget(
        fa.SaltAndPepperNoiseAttack(steps=50, channel_axis=1),
        None,
        uses_grad=True,
        stochastic_attack=True,
    ),
    AttackTestTarget(
        fa.LinearSearchBlendedUniformNoiseAttack(steps=50), None, stochastic_attack=True
    ),
    AttackTestTarget(
        fa.L2AdditiveGaussianNoiseAttack(), 3000.0, stochastic_attack=True
    ),
    AttackTestTarget(
        fa.L2ClippingAwareAdditiveGaussianNoiseAttack(), 500.0, stochastic_attack=True
    ),
    AttackTestTarget(fa.LinfAdditiveUniformNoiseAttack(), 10.0, stochastic_attack=True),
    AttackTestTarget(
        fa.L2RepeatedAdditiveGaussianNoiseAttack(check_trivial=False),
        1000.0,
        stochastic_attack=True,
    ),
    AttackTestTarget(
        fa.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(check_trivial=False),
        200.0,
        stochastic_attack=True,
    ),
    AttackTestTarget(
        fa.L2RepeatedAdditiveGaussianNoiseAttack(), 1000.0, stochastic_attack=True
    ),
    AttackTestTarget(
        fa.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(),
        200.0,
        stochastic_attack=True,
    ),
    AttackTestTarget(
        fa.L2RepeatedAdditiveUniformNoiseAttack(), 1000.0, stochastic_attack=True
    ),
    AttackTestTarget(
        fa.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(),
        200.0,
        stochastic_attack=True,
    ),
    AttackTestTarget(
        fa.LinfRepeatedAdditiveUniformNoiseAttack(), 3.0, stochastic_attack=True
    ),
]


@pytest.mark.parametrize("attack_test_target", attacks, ids=get_attack_id)
def test_untargeted_attacks(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_test_target: AttackTestTarget,
) -> None:

    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks
    if attack_test_target.requires_real_model and not real:
        pytest.skip()
    if attack_test_target.requires_low_dimensional_input and not low_dimensional_input:
        pytest.skip()
    if isinstance(x, ep.NumPyTensor) and attack_test_target.uses_grad:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    acc = fbn.accuracy(fmodel, x, y)
    assert acc > 0

    # repeat stochastic attacks three times before we mark them as failed
    attack_repetitions = 3 if attack_test_target.stochastic_attack else 1

    for _ in range(attack_repetitions):
        advs, _, _ = attack_test_target.attack(
            fmodel, x, y, epsilons=attack_test_target.epsilon
        )
        adv_acc = fbn.accuracy(fmodel, advs, y)
        if adv_acc < acc:
            break
    assert adv_acc < acc


targeted_attacks: List[AttackTestTarget] = [
    AttackTestTarget(
        fa.L2CarliniWagnerAttack(
            binary_search_steps=2, steps=100, stepsize=0.05, initial_const=1e1
        ),
        uses_grad=True,
    ),
    AttackTestTarget(fa.DDNAttack(init_epsilon=2.0, steps=20), uses_grad=True),
    # TODO: targeted EADAttack currently fails repeatedly on MobileNetv2
    AttackTestTarget(
        fa.EADAttack(
            binary_search_steps=3,
            steps=20,
            abort_early=True,
            regularization=0,
            initial_const=1e1,
        ),
        uses_grad=True,
    ),
    AttackTestTarget(
        fa.GenAttack(steps=100, population=6, reduced_dims=(8, 8)),
        epsilon=0.5,
        requires_real_model=True,
        requires_low_dimensional_input=True,
    ),
    AttackTestTarget(fa.PGD(), Linf(1.0), uses_grad=True),
    AttackTestTarget(fa.L2PGD(), L2(50.0), uses_grad=True),
    AttackTestTarget(fa.L1PGD(), 5000.0, uses_grad=True),
    AttackTestTarget(
        fa.LinfBasicIterativeAttack(abs_stepsize=0.2), Linf(1.0), uses_grad=True
    ),
    AttackTestTarget(fa.L2BasicIterativeAttack(), L2(50.0), uses_grad=True),
    AttackTestTarget(fa.L1BasicIterativeAttack(), 5000.0, uses_grad=True),
    AttackTestTarget(fa.AdamPGD(), Linf(1.0), uses_grad=True),
    AttackTestTarget(fa.L2AdamPGD(), L2(50.0), uses_grad=True),
    AttackTestTarget(fa.L1AdamPGD(), 5000.0, uses_grad=True),
    AttackTestTarget(
        fa.LinfAdamBasicIterativeAttack(abs_stepsize=0.2), Linf(1.0), uses_grad=True
    ),
    AttackTestTarget(fa.L2AdamBasicIterativeAttack(), L2(50.0), uses_grad=True),
    AttackTestTarget(fa.L1AdamBasicIterativeAttack(), 5000.0, uses_grad=True),
    AttackTestTarget(fa.SparseL1DescentAttack(), 5000.0, uses_grad=True),
]


@pytest.mark.parametrize("attack_test_target", targeted_attacks, ids=get_attack_id)
def test_targeted_attacks(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_test_target: AttackTestTarget,
) -> None:

    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks
    if attack_test_target.requires_real_model and not real:
        pytest.skip()
    if attack_test_target.requires_low_dimensional_input and not low_dimensional_input:
        pytest.skip()

    if isinstance(x, ep.NumPyTensor) and attack_test_target.uses_grad:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    num_classes = fmodel(x).shape[-1]
    target_classes = (y + 1) % num_classes
    criterion = fbn.TargetedMisclassification(target_classes)
    adv_before_attack = criterion(x, fmodel(x))
    assert not adv_before_attack.all()

    asr = adv_before_attack.sum().item()

    # repeat stochastic attacks three times before we mark them as failed
    attack_repetitions = 3 if attack_test_target.stochastic_attack else 1

    for _ in range(attack_repetitions):
        advs, _, _ = attack_test_target.attack(
            fmodel, x, criterion, epsilons=attack_test_target.epsilon
        )
        adv_after_attack = criterion(advs, fmodel(advs))
        adv_asr = adv_after_attack.sum().item()
        if adv_asr > asr:
            break
    assert adv_asr > asr
