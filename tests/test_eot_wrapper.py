import pytest

import eagerpy as ep

from foolbox import accuracy
from foolbox.attacks import (
    LinfBasicIterativeAttack,
    L1BasicIterativeAttack,
    L2BasicIterativeAttack,
)
from foolbox.models import ExpectationOverTransformationWrapper
from foolbox.types import L2, Linf

from conftest import ModeAndDataAndDescription


def test_eot_wrapper(
    fmodel_and_data_ext_for_attacks: ModeAndDataAndDescription,
) -> None:

    (fmodel, x, y), real, low_dimensional_input = fmodel_and_data_ext_for_attacks

    if isinstance(x, ep.NumPyTensor):
        pytest.skip()

    # test clean accuracy when wrapping EoT
    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))
    acc = accuracy(fmodel, x, y)

    rand_model = ExpectationOverTransformationWrapper(fmodel, n_steps=4)
    rand_acc = accuracy(rand_model, x, y)
    assert acc - rand_acc == 0

    # test with base attacks
    # (accuracy should not change, since fmodel is not random)
    attacks = (
        L1BasicIterativeAttack(),
        L2BasicIterativeAttack(),
        LinfBasicIterativeAttack(),
    )
    epsilons = (5000.0, L2(50.0), Linf(1.0))

    for attack, eps in zip(attacks, epsilons):

        # acc on standard model
        advs, _, _ = attack(fmodel, x, y, epsilons=eps)
        adv_acc = accuracy(fmodel, advs, y)

        # acc on eot model
        advs, _, _ = attack(rand_model, x, y, epsilons=eps)
        r_adv_acc = accuracy(rand_model, advs, y)
        assert adv_acc - r_adv_acc == 0
