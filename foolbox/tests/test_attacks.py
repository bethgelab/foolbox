import sys
if sys.version_info > (3, 2):
    from unittest.mock import Mock
else:
    # for Python2.7 compatibility
    from mock import Mock

import pytest

from foolbox import attacks
from foolbox import Adversarial


# def test_abstract_attack():
#     with pytest.raises(TypeError):
#         attacks.Attack()


def test_base_init():
    assert attacks.FGSM() is not None
    assert attacks.FGSM(Mock()) is not None
    assert attacks.FGSM(None, None) is not None
    assert attacks.FGSM(Mock(), Mock()) is not None


def test_aliases():
    assert attacks.GradientSignAttack == attacks.FGSM


def test_base_attack(model, criterion, image, label):
    attack = attacks.FGSM(model, criterion)
    assert attack.name() == 'GradientSignAttack'

    with pytest.raises(ValueError):
        attack(image)

    with pytest.raises(TypeError):
        attack(label=label)

    wrong_label = label + 1

    adv = attack(image, label=label)
    assert adv is None
    adv = attack(image, label=wrong_label)
    assert adv.shape == image.shape
    adv = attack(image, label=wrong_label, unpack=False)
    assert adv.perturbed.shape == image.shape

    adv = Adversarial(model, criterion, image, wrong_label)
    adv = attack(adv)
    assert adv.shape == image.shape

    adv = Adversarial(model, criterion, image, wrong_label)
    with pytest.raises(ValueError):
        attack(adv, label=wrong_label)

    attack = attacks.FGSM()
    with pytest.raises(ValueError):
        attack(image, label=wrong_label)


def test_early_stopping(bn_model, bn_criterion, bn_image, bn_label):
    attack = attacks.FGSM()

    model = bn_model
    criterion = bn_criterion
    image = bn_image
    label = bn_label

    wrong_label = label + 1
    adv = Adversarial(model, criterion, image, wrong_label)
    attack(adv)
    assert adv.distance.value == 0
    assert not adv.reached_threshold()  # because no threshold specified

    adv = Adversarial(model, criterion, image, wrong_label, threshold=1e10)
    attack(adv)
    assert adv.distance.value == 0
    assert adv.reached_threshold()

    adv = Adversarial(model, criterion, image, label)
    attack(adv)
    assert adv.distance.value > 0
    assert not adv.reached_threshold()  # because no threshold specified

    c = adv._total_prediction_calls
    d = adv.distance.value
    large_d = 10 * d
    small_d = d / 2

    adv = Adversarial(model, criterion, image, label,
                      threshold=adv._distance(value=large_d))
    attack(adv)
    assert 0 < adv.distance.value <= large_d
    assert adv.reached_threshold()
    assert adv._total_prediction_calls < c

    adv = Adversarial(model, criterion, image, label,
                      threshold=large_d)
    attack(adv)
    assert 0 < adv.distance.value <= large_d
    assert adv.reached_threshold()
    assert adv._total_prediction_calls < c

    adv = Adversarial(model, criterion, image, label,
                      threshold=small_d)
    attack(adv)
    assert small_d < adv.distance.value <= large_d
    assert not adv.reached_threshold()
    assert adv._total_prediction_calls == c
    assert adv.distance.value == d

    adv = Adversarial(model, criterion, image, label,
                      threshold=adv._distance(value=large_d))
    attack(adv)
    assert adv.reached_threshold()
    c = adv._total_prediction_calls
    attack(adv)
    assert adv._total_prediction_calls == c  # no new calls
