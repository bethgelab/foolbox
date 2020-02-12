from typing import Tuple
import eagerpy as ep

import foolbox as fbn
from foolbox.devutils import flatten


def test_binarization_attack(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
) -> None:

    fmodel, x, y = fmodel_and_data
    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    fmodel = fbn.models.ThresholdingWrapper(fmodel, threshold=0.5)

    attack = fbn.attacks.BinarySearchContrastReductionAttack(target=0)
    advs = attack(fmodel, x, y)

    attack2 = fbn.attacks.BinarizationRefinementAttack(
        threshold=0.5, included_in="upper"
    )
    advs2 = attack2(fmodel, x, y, starting_points=advs)

    assert (fmodel(advs).argmax(axis=-1) == fmodel(advs2).argmax(axis=-1)).all()

    assert (
        flatten(advs2 - x).norms.l2(axis=-1) <= flatten(advs - x).norms.l2(axis=-1)
    ).all()
    assert (
        flatten(advs2 - x).norms.l2(axis=-1) < flatten(advs - x).norms.l2(axis=-1)
    ).any()
