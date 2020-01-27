import pytest
import foolbox.ext.native as fbn
from typing import List, Tuple, Type, Dict, Any

attacks: List[Tuple[Type[fbn.attacks.Attack], Dict[str, Any], Dict[str, Any]]] = [
    (fbn.attacks.InversionAttack, {}, {})
]


@pytest.mark.parametrize("attack", attacks)
def test_callable(fmodel_and_data, attack):
    fmodel, x, y = fmodel_and_data
    attack, init_kwargs, call_kwargs = attack
    attack = attack(fmodel, **init_kwargs)
    advs = attack(x, y, **call_kwargs)
    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
