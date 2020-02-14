from typing import Tuple
import eagerpy as ep

from foolbox import accuracy, Model
from foolbox.models import ThresholdingWrapper
from foolbox.devutils import flatten
from foolbox.attacks import BinarySearchContrastReductionAttack
from foolbox.attacks import BinarizationRefinementAttack


def test_binarization_attack(
    fmodel_and_data_ext_for_attacks: Tuple[Tuple[Model, ep.Tensor, ep.Tensor], bool],
) -> None:

    # get a model with thresholding
    (fmodel, x, y), _ = fmodel_and_data_ext_for_attacks
    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    fmodel = ThresholdingWrapper(fmodel, threshold=0.5)
    acc = accuracy(fmodel, x, y)
    assert acc > 0

    # find some adversarials and check that they are non-trivial
    attack = BinarySearchContrastReductionAttack(target=0)
    advs, _, _ = attack(fmodel, x, y, epsilons=None)
    assert accuracy(fmodel, advs, y) < acc

    # run the refinement attack
    attack2 = BinarizationRefinementAttack(threshold=0.5, included_in="upper")
    advs2, _, _ = attack2(fmodel, x, y, starting_points=advs, epsilons=None)

    # make sure the predicted classes didn't change
    assert (fmodel(advs).argmax(axis=-1) == fmodel(advs2).argmax(axis=-1)).all()

    # make sure the perturbations didn't get larger and some got smaller
    norms1 = flatten(advs - x).norms.l2(axis=-1)
    norms2 = flatten(advs2 - x).norms.l2(axis=-1)
    assert (norms2 <= norms1).all()
    assert (norms2 < norms1).any()
