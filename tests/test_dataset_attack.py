import pytest

import foolbox as fbn

from conftest import ModeAndDataAndDescription


def test_dataset_attack(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:

    (fmodel, x, y), _, _ = fmodel_and_data_ext_for_attacks
    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    attack = fbn.attacks.DatasetAttack()
    attack.feed(fmodel, x)

    assert fbn.accuracy(fmodel, x, y) > 0

    advs, _, success = attack(fmodel, x, y, epsilons=None)
    assert success.shape == (len(x),)
    assert success.all()
    assert fbn.accuracy(fmodel, advs, y) == 0

    with pytest.raises(ValueError, match="unknown distance"):
        attack(fmodel, x, y, epsilons=[500.0, 1000.0])
    attack = fbn.attacks.DatasetAttack(distance=fbn.distances.l2)
    attack.feed(fmodel, x)
    advss, _, success = attack(fmodel, x, y, epsilons=[500.0, 1000.0])
    assert success.shape == (2, len(x))
    assert success.all()
    assert fbn.accuracy(fmodel, advss[0], y) == 0
    assert fbn.accuracy(fmodel, advss[1], y) == 0

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        attack(fmodel, x, y, epsilons=None, invalid=True)
