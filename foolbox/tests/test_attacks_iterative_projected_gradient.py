import pytest
import numpy as np

from foolbox.attacks import LinfinityBasicIterativeAttack
from foolbox.attacks import L1BasicIterativeAttack
from foolbox.attacks import L2BasicIterativeAttack
from foolbox.attacks import ProjectedGradientDescentAttack
from foolbox.attacks import RandomStartProjectedGradientDescentAttack
from foolbox.attacks import MomentumIterativeAttack

Attacks = [
    LinfinityBasicIterativeAttack,
    L1BasicIterativeAttack,
    L2BasicIterativeAttack,
    ProjectedGradientDescentAttack,
    RandomStartProjectedGradientDescentAttack,
    MomentumIterativeAttack,
]


@pytest.mark.parametrize('Attack', Attacks)
def test_attack(Attack, bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_targeted_attack(Attack, bn_targeted_adversarial):
    adv = bn_targeted_adversarial
    attack = Attack()
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_gl(Attack, gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.image is None
    assert adv.distance.value == np.inf
