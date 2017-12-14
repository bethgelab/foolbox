import numpy as np
import pytest
from foolbox.attacks import GradientAttack
from foolbox.attacks import GradientSignAttack


@pytest.mark.parametrize('attack', [GradientAttack(), GradientSignAttack()])
def test_attack(bn_adversarial, attack):
    adv = bn_adversarial
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.image is None
    assert adv.distance.value == np.inf
