from typing import List, Tuple
import pytest
import eagerpy as ep
import foolbox as fbn

L2 = fbn.types.L2
Linf = fbn.types.Linf


def test_ead_init_raises() -> None:
    with pytest.raises(ValueError, match="invalid decision rule"):
        fbn.attacks.EADAttack(binary_search_steps=3, steps=20, decision_rule="L2")  # type: ignore


def test_deepfool_init_raises() -> None:
    with pytest.raises(ValueError, match="expected loss to be"):
        fbn.attacks.L2DeepFoolAttack(loss="invalid")  # type: ignore


def test_blended_noise_attack_run_warns(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]
) -> None:
    fmodel, x, y = fmodel_and_data
    attack = fbn.attacks.LinearSearchBlendedUniformNoiseAttack(directions=1)
    attack.run(fmodel, x, y)


def test_boundary_attack_run_raises(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]
) -> None:
    fmodel, x, y = fmodel_and_data
    with pytest.raises(ValueError, match="starting_points are not adversarial"):
        attack = fbn.attacks.BoundaryAttack()
        attack.run(fmodel, x, y, starting_points=x)

    with pytest.raises(ValueError, match="init_attack failed for"):
        attack = fbn.attacks.BoundaryAttack(
            init_attack=fbn.attacks.DDNAttack(init_epsilon=0.0, steps=1)
        )
        attack.run(fmodel, x, y)


def test_newtonfool_run_raises(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]
) -> None:
    fmodel, x, y = fmodel_and_data
    with pytest.raises(ValueError, match="unsupported criterion"):
        attack = fbn.attacks.NewtonFoolAttack()
        attack.run(fmodel, x, fbn.TargetedMisclassification(y))

    with pytest.raises(ValueError, match="expected labels to have shape"):
        attack = fbn.attacks.NewtonFoolAttack(steps=10)
        attack.run(fmodel, x, ep.concatenate((y, y), 0))


def test_fgsm_run_raises(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]
) -> None:
    fmodel, x, y = fmodel_and_data
    with pytest.raises(ValueError, match="unsupported criterion"):
        attack = fbn.attacks.FGSM()
        attack.run(fmodel, x, fbn.TargetedMisclassification(y), epsilon=1000)


def test_vat_run_raises(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]
) -> None:
    fmodel, x, y = fmodel_and_data
    with pytest.raises(ValueError, match="unsupported criterion"):
        attack = fbn.attacks.VirtualAdversarialAttack(steps=10)
        attack.run(fmodel, x, fbn.TargetedMisclassification(y), epsilon=1.0)

    with pytest.raises(ValueError, match="expected labels to have shape"):
        attack = fbn.attacks.VirtualAdversarialAttack(steps=10)
        attack.run(fmodel, x, ep.concatenate((y, y), 0), epsilon=1.0)


def test_blended_noise_init_raises() -> None:
    with pytest.raises(ValueError, match="directions must be larger than 0"):
        fbn.attacks.LinearSearchBlendedUniformNoiseAttack(steps=50, directions=0)


def test_blur_run_raises(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]
) -> None:
    fmodel, x, y = fmodel_and_data
    with pytest.raises(ValueError, match="to be 1 or 3"):
        attack = fbn.attacks.GaussianBlurAttack(steps=10, channel_axis=2)
        attack.run(fmodel, x, y)


targeted_attacks_raises_exception: List[Tuple[fbn.Attack, bool]] = [
    (fbn.attacks.EADAttack(), True),
    (fbn.attacks.DDNAttack(), True),
    (fbn.attacks.L2CarliniWagnerAttack(), True),
]


@pytest.mark.parametrize(
    "attack_exception_text_and_grad", targeted_attacks_raises_exception
)
def test_targeted_attacks_call_raises_exception(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
    attack_exception_text_and_grad: Tuple[fbn.Attack, bool],
) -> None:

    attack, attack_uses_grad = attack_exception_text_and_grad
    fmodel, x, y = fmodel_and_data

    if isinstance(x, ep.NumPyTensor) and attack_uses_grad:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    num_classes = fmodel(x).shape[-1]
    target_classes = (y + 1) % num_classes
    invalid_target_classes = ep.concatenate((target_classes, target_classes), 0)
    invalid_targeted_criterion = fbn.TargetedMisclassification(invalid_target_classes)

    class DummyCriterion(fbn.Criterion):
        """Criterion without any functionality which is just meant to be
        rejected by the attacks
        """

        def __repr__(self) -> str:
            return ""

        def __call__(
            self, perturbed: fbn.criteria.T, outputs: fbn.criteria.T
        ) -> fbn.criteria.T:
            return perturbed

    invalid_criterion = DummyCriterion()

    # check if targeted attack criterion with invalid number of classes is rejected
    with pytest.raises(ValueError):
        attack(fmodel, x, invalid_targeted_criterion, epsilons=1000.0)

    # check if only the two valid criteria are accepted
    with pytest.raises(ValueError):
        attack(fmodel, x, invalid_criterion, epsilons=1000.0)
