import pytest
import numpy as np

from foolbox.attacks import GradientAttack
from foolbox.attacks import BinarizationRefinementAttack


def test_attack(binarized_bn_adversarial):
    adv = binarized_bn_adversarial

    attack = GradientAttack()
    attack(adv)
    v1 = adv.distance.value

    attack = BinarizationRefinementAttack()
    attack(adv)
    v2 = adv.distance.value

    assert v2 < v1 < np.inf

    o = adv.original_image
    x = adv.image
    d = x[x != o]
    np.testing.assert_allclose(d, 0.5)


def test_attack_fail(bn_adversarial):
    adv = bn_adversarial

    attack = GradientAttack()
    attack(adv)
    assert adv is not None

    attack = BinarizationRefinementAttack()
    with pytest.raises(AssertionError) as e:
        attack(adv)
    assert 'thresholding does not match' in str(e.value)


def test_attack_noinit(binarized_bn_adversarial):
    adv = binarized_bn_adversarial
    assert adv.image is None

    attack = BinarizationRefinementAttack()
    attack(adv)
    assert adv.image is None


def test_attack_sp(binarized_bn_adversarial):
    adv = binarized_bn_adversarial

    attack = GradientAttack()
    attack(adv)
    v1 = adv.distance.value

    attack = BinarizationRefinementAttack(adv._model)
    adv = attack(adv.original_image, adv.original_class,
                 starting_point=adv.image, unpack=False)
    v2 = adv.distance.value

    assert v2 < v1 < np.inf

    o = adv.original_image
    x = adv.image
    d = x[x != o]
    np.testing.assert_allclose(d, 0.5)


def test_attack2(binarized2_bn_adversarial):
    adv = binarized2_bn_adversarial

    attack = GradientAttack()
    attack(adv)
    v1 = adv.distance.value

    attack = BinarizationRefinementAttack()
    attack(adv, included_in='lower')
    v2 = adv.distance.value

    assert v2 < v1 < np.inf

    o = adv.original_image
    x = adv.image
    d = x[x != o]
    np.testing.assert_allclose(d, 0.5)


def test_attack_wrong_arg(binarized_bn_adversarial):
    adv = binarized_bn_adversarial

    attack = GradientAttack()
    attack(adv)

    attack = BinarizationRefinementAttack()
    with pytest.raises(ValueError):
        attack(adv, included_in='blabla')
