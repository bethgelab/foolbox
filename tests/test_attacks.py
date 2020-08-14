from typing import List, Tuple, Optional
import pytest
import eagerpy as ep

import foolbox as fbn
import foolbox.attacks as fa
from foolbox.gradient_estimators import es_gradient_estimator

L2 = fbn.types.L2
Linf = fbn.types.Linf


FGSM_GE = es_gradient_estimator(
    fa.FGSM, samples=100, sigma=0.03, bounds=(0, 1), clip=True
)


def get_attack_id(x: Tuple[fbn.Attack, bool, bool]) -> str:
    return repr(x[0])


# attack, eps / None, attack_uses_grad, requires_real_model
attacks: List[Tuple[fbn.Attack, Optional[float], bool, bool]] = [
    (fa.DDNAttack(init_epsilon=2.0), None, True, False),
    (fa.InversionAttack(), None, False, False),
    (
        fa.InversionAttack(distance=fbn.distances.l2).repeat(3).repeat(2),
        None,
        False,
        False,
    ),
    (fa.L2ContrastReductionAttack(), L2(100.0), False, False),
    (fa.L2ContrastReductionAttack().repeat(3), 100.0, False, False),
    (
        fa.BinarySearchContrastReductionAttack(binary_search_steps=15),
        None,
        False,
        False,
    ),
    (fa.LinearSearchContrastReductionAttack(steps=20), None, False, False),
    (fa.L2CarliniWagnerAttack(binary_search_steps=11, steps=5), None, True, False),
    (
        fa.L2CarliniWagnerAttack(binary_search_steps=3, steps=20, confidence=2.0),
        None,
        True,
        False,
    ),
    (
        fa.EADAttack(binary_search_steps=10, steps=20, regularization=0),
        None,
        True,
        False,
    ),
    (
        fa.EADAttack(
            binary_search_steps=10, steps=20, regularization=0, confidence=2.0
        ),
        None,
        True,
        False,
    ),
    (
        fa.EADAttack(
            binary_search_steps=3, steps=20, decision_rule="L1", regularization=0
        ),
        None,
        True,
        False,
    ),
    (fa.NewtonFoolAttack(steps=20), None, True, False),
    (fa.VirtualAdversarialAttack(steps=50, xi=1), 10, True, False),
    (fa.PGD(), Linf(1.0), True, False),
    (fa.L2PGD(), L2(50.0), True, False),
    (fa.L1PGD(), 5000.0, True, False),
    (fa.LinfBasicIterativeAttack(abs_stepsize=0.2), Linf(1.0), True, False),
    (fa.L2BasicIterativeAttack(), L2(50.0), True, False),
    (fa.L1BasicIterativeAttack(), 5000.0, True, False),
    (fa.SparseL1DescentAttack(), 5000.0, True, False),
    (fa.FGSM(), Linf(100.0), True, False),
    (FGSM_GE(), Linf(100.0), False, False),
    (fa.FGM(), L2(100.0), True, False),
    (fa.L1FastGradientAttack(), 5000.0, True, False),
    (fa.GaussianBlurAttack(steps=10), None, True, True),
    (fa.GaussianBlurAttack(steps=10, max_sigma=224.0), None, True, True),
    (fa.L2DeepFoolAttack(steps=50, loss="logits"), None, True, False),
    (fa.L2DeepFoolAttack(steps=50, loss="crossentropy"), None, True, False),
    (fa.LinfDeepFoolAttack(steps=50), None, True, False),
    (fa.BoundaryAttack(steps=50), None, False, False),
    (
        fa.BoundaryAttack(
            steps=110,
            init_attack=fa.LinearSearchBlendedUniformNoiseAttack(steps=50),
            update_stats_every_k=1,
        ),
        None,
        False,
        False,
    ),
    (fa.SaltAndPepperNoiseAttack(steps=50), None, True, False),
    (fa.SaltAndPepperNoiseAttack(steps=50, channel_axis=1), None, True, False),
    (fa.LinearSearchBlendedUniformNoiseAttack(steps=50), None, False, False),
    (fa.L2AdditiveGaussianNoiseAttack(), 2500.0, False, False),
    (fa.L2ClippingAwareAdditiveGaussianNoiseAttack(), 500.0, False, False),
    (fa.LinfAdditiveUniformNoiseAttack(), 10.0, False, False),
    (
        fa.L2RepeatedAdditiveGaussianNoiseAttack(check_trivial=False),
        1000.0,
        False,
        False,
    ),
    (
        fa.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(check_trivial=False),
        200.0,
        False,
        False,
    ),
    (fa.L2RepeatedAdditiveGaussianNoiseAttack(), 1000.0, False, False),
    (fa.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(), 200.0, False, False),
    (fa.L2RepeatedAdditiveUniformNoiseAttack(), 1000.0, False, False),
    (fa.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(), 200.0, False, False),
    (fa.LinfRepeatedAdditiveUniformNoiseAttack(), 3.0, False, False),
]


@pytest.mark.parametrize("attack_eps_grad_real", attacks, ids=get_attack_id)
def test_untargeted_attacks(
    fmodel_and_data_ext_for_attacks: Tuple[
        Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool
    ],
    attack_eps_grad_real: Tuple[fbn.Attack, Optional[float], bool, bool],
) -> None:

    attack, eps, attack_uses_grad, requires_real_model = attack_eps_grad_real
    (fmodel, x, y), real = fmodel_and_data_ext_for_attacks
    if requires_real_model and not real:
        pytest.skip()

    if isinstance(x, ep.NumPyTensor) and attack_uses_grad:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    acc = fbn.accuracy(fmodel, x, y)
    assert acc > 0

    advs, _, _ = attack(fmodel, x, y, epsilons=eps)
    assert fbn.accuracy(fmodel, advs, y) < acc


targeted_attacks: List[Tuple[fbn.Attack, Optional[float], bool, bool]] = [
    (
        fa.L2CarliniWagnerAttack(binary_search_steps=3, steps=20, initial_const=1e1),
        None,
        True,
        False,
    ),
    (fa.DDNAttack(init_epsilon=2.0, steps=20), None, True, False),
    # TODO: targeted EADAttack currently fails repeatedly on MobileNetv2
    (
        fa.EADAttack(
            binary_search_steps=3,
            steps=20,
            abort_early=True,
            regularization=0,
            initial_const=1e1,
        ),
        None,
        True,
        False,
    ),
    (fa.GenAttack(steps=100, population=6, reduced_dims=(7, 7)), 0.5, False, True),
]


@pytest.mark.parametrize("attack_eps_grad_real", targeted_attacks, ids=get_attack_id)
def test_targeted_attacks(
    fmodel_and_data_ext_for_attacks: Tuple[
        Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool
    ],
    attack_eps_grad_real: Tuple[fbn.Attack, Optional[float], bool, bool],
) -> None:

    attack, eps, attack_uses_grad, requires_real_model = attack_eps_grad_real
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
    adv_before_attack = criterion(x, fmodel(x))
    assert not adv_before_attack.all()

    advs, _, _ = attack(fmodel, x, criterion, epsilons=eps)
    adv_after_attack = criterion(advs, fmodel(advs))
    assert adv_after_attack.sum().item() > adv_before_attack.sum().item()
