import numpy as np

from foolbox.attacks import IterativeGradientAttack as Attack


def test_attack(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, epsilons=10, steps=5)
    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv, epsilons=10, steps=5)
    assert adv.image is None
    assert adv.distance.value == np.inf
