import numpy as np

from foolbox.attacks import LocalSearchAttack as Attack


def test_attack(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, d=1, t=10)
    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv, d=1, t=10)
    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_targeted_attack(bn_targeted_adversarial):
    adv = bn_targeted_adversarial
    attack = Attack()
    attack(adv, d=1)
    assert adv.image is None
    assert adv.distance.value == np.inf
