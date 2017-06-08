from unittest.mock import Mock

import pytest

from foolbox import attacks
from foolbox import Adversarial


def test_abstract_attack():
    with pytest.raises(TypeError):
        attacks.Attack()


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
        attack(image=image)

    with pytest.raises(ValueError):
        attack(label=label)

    adv = attack(image=image, label=label)
    assert adv.shape == image.shape
    adv = attack(image=image, label=label, unpack=False)
    assert adv.get().shape == image.shape

    adv = Adversarial(model, criterion, image, label)
    adv = attack(adv)
    assert adv.shape == image.shape

    with pytest.raises(ValueError):
        attack(adv, label=label)

    attack = attacks.FGSM()
    with pytest.raises(ValueError):
        attack(image=image, label=label)
