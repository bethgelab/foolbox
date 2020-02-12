from typing import List, Tuple
import pytest
import eagerpy as ep

import foolbox as fbn

L2 = fbn.types.L2
Linf = fbn.types.Linf


def test_ead_init_raises() -> None:
    with pytest.raises(ValueError, match="invalid decision rule"):
        fbn.attacks.EADAttack(binary_search_steps=3, steps=20, decision_rule="L2")  # type: ignore


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
