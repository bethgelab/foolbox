import pytest
import foolbox.ext.native as fbn
from typing import List, Tuple, Type, Dict, Any

attacks: List[Tuple[Type[fbn.attacks.Attack], Dict[str, Any], Dict[str, Any]]] = [
    (fbn.attacks.InversionAttack, {}, {}),
    # (fbn.attacks.L2AdditiveGaussianNoiseAttack, {}, dict(epsilon=2000.0)),  # currently, this epsilon is absolute
    # (fbn.attacks.L2AdditiveUniformNoiseAttack, {}, dict(epsilon=2000.0)),  # (l2 norm of data between the bounds)
    # (fbn.attacks.LinfAdditiveUniformNoiseAttack, {}, dict(epsilon=100.0)),
    # (fbn.attacks.L2BasicIterativeAttack, {}, dict(epsilon=2000.0)),
]


@pytest.mark.parametrize("attack", attacks)
def test_callable(fmodel_and_data_for_attacks, attack):
    fmodel, x, y = fmodel_and_data_for_attacks
    attack, init_kwargs, call_kwargs = attack
    attack = attack(fmodel, **init_kwargs)
    advs = attack(x, y, **call_kwargs)
    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
