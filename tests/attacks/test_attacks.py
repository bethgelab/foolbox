from typing import List, Tuple
import pytest
import eagerpy as ep

import foolbox.ext.native as fbn

attacks: List[fbn.Attack] = [fbn.attacks.InversionAttack()]


@pytest.mark.parametrize("attack", attacks)
def test_init_and_call(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor], attack: fbn.Attack
) -> None:
    fmodel, x, y = fmodel_and_data
    advs = attack(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
