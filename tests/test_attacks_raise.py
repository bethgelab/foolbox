from typing import List, Tuple, Any
import pytest
import eagerpy as ep
import foolbox as fbn

from conftest import ModeAndDataAndDescription

L2 = fbn.types.L2
Linf = fbn.types.Linf


def test_ead_init_raises() -> None:
    with pytest.raises(ValueError, match="invalid decision rule"):
        fbn.attacks.EADAttack(binary_search_steps=3, steps=20, decision_rule="invalid")  # type: ignore


def test_genattack_numpy(request: Any) -> None:
    class Model:
        def __call__(self, inputs: Any) -> Any:
            return inputs.mean(axis=(2, 3))

    model = Model()
    with pytest.raises(ValueError):
        fbn.NumPyModel(model, bounds=(0, 1), data_format="foo")

    fmodel = fbn.NumPyModel(model, bounds=(0, 1))
    x, y = ep.astensors(
        *fbn.samples(
            fmodel, dataset="imagenet", batchsize=16, data_format="channels_first"
        )
    )

    with pytest.raises(ValueError, match="data_format"):
        fbn.attacks.GenAttack(reduced_dims=(2, 2)).run(
            fmodel, x, fbn.TargetedMisclassification(y), epsilon=0.3
        )

    with pytest.raises(ValueError, match="channel_axis"):
        fbn.attacks.GenAttack(channel_axis=2, reduced_dims=(2, 2)).run(
            fmodel, x, fbn.TargetedMisclassification(y), epsilon=0.3
        )


def test_deepfool_run_raises(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    attack = fbn.attacks.L2DeepFoolAttack(loss="invalid")  # type: ignore
    with pytest.raises(ValueError, match="expected loss to"):
        attack.run(fmodel, x, y)


def test_blended_noise_attack_run_warns(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    attack = fbn.attacks.LinearSearchBlendedUniformNoiseAttack(directions=1)
    attack.run(fmodel, x, y)


def test_boundary_attack_run_raises(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks

    with pytest.raises(ValueError, match="starting_points are not adversarial"):
        attack = fbn.attacks.BoundaryAttack()
        attack.run(fmodel, x, y, starting_points=x)

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()
    with pytest.raises(ValueError, match="init_attack failed for"):
        attack = fbn.attacks.BoundaryAttack(
            init_attack=fbn.attacks.DDNAttack(init_epsilon=0.0, steps=1)
        )
        attack.run(fmodel, x, y)


def test_newtonfool_run_raises(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    with pytest.raises(ValueError, match="unsupported criterion"):
        attack = fbn.attacks.NewtonFoolAttack()
        attack.run(fmodel, x, fbn.TargetedMisclassification(y))

    with pytest.raises(ValueError, match="expected labels to have shape"):
        attack = fbn.attacks.NewtonFoolAttack(steps=10)
        attack.run(fmodel, x, ep.concatenate((y, y), 0))


def test_fgsm_run_raises(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    with pytest.raises(ValueError, match="unsupported criterion"):
        attack = fbn.attacks.FGSM()
        attack.run(fmodel, x, fbn.TargetedMisclassification(y), epsilon=1000)


def test_vat_run_raises(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

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
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    with pytest.raises(ValueError, match="to be 1 or 3"):
        attack = fbn.attacks.GaussianBlurAttack(steps=10, channel_axis=2)
        attack.run(fmodel, x, y)


def test_blur_numpy(request: Any) -> None:
    class Model:
        def __call__(self, inputs: Any) -> Any:
            return inputs.mean(axis=(2, 3))

    model = Model()
    with pytest.raises(ValueError):
        fbn.NumPyModel(model, bounds=(0, 1), data_format="foo")

    fmodel = fbn.NumPyModel(model, bounds=(0, 1))
    x, y = ep.astensors(
        *fbn.samples(
            fmodel, dataset="imagenet", batchsize=16, data_format="channels_first"
        )
    )
    with pytest.raises(ValueError, match="data_format"):
        fbn.attacks.GaussianBlurAttack()(fmodel, x, y, epsilons=None)


def test_dataset_attack_raises(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks

    attack = fbn.attacks.DatasetAttack()

    # that that running before feed fails properly
    with pytest.raises(ValueError, match="feed"):
        attack.run(fmodel, x, y)

    attack.feed(fmodel, x)
    attack.run(fmodel, x, y)
    assert attack.inputs is not None
    n = len(attack.inputs)

    # test that feed() after run works
    attack.feed(fmodel, x)
    attack.run(fmodel, x, y)
    assert len(attack.inputs) > n


targeted_attacks_raises_exception: List[Tuple[fbn.Attack, bool]] = [
    (fbn.attacks.EADAttack(), True),
    (fbn.attacks.DDNAttack(), True),
    (fbn.attacks.L2CarliniWagnerAttack(), True),
    (fbn.attacks.GenAttack(), False),
]


@pytest.mark.parametrize(
    "attack_exception_text_and_grad", targeted_attacks_raises_exception
)
def test_targeted_attacks_call_raises_exception(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
    attack_exception_text_and_grad: Tuple[fbn.Attack, bool],
) -> None:

    attack, attack_uses_grad = attack_exception_text_and_grad
    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks

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
