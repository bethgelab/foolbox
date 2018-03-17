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
    assert adv.image.shape == image.shape

    adv = Adversarial(model, criterion, image, wrong_label)
    adv = attack(adv)
    assert adv.shape == image.shape

    adv = Adversarial(model, criterion, image, wrong_label)
    with pytest.raises(ValueError):
        attack(adv, label=wrong_label)

    attack = attacks.FGSM()
    with pytest.raises(ValueError):
        attack(image, label=wrong_label)
