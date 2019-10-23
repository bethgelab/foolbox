import numpy as np

from foolbox import set_seeds
from foolbox.v1.attacks import LocalSearchAttack as Attack


def test_attack(bn_adversarial):
    set_seeds(22)
    adv = bn_adversarial
    attack = Attack()
    attack(adv, d=1, t=10)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_adversarial):
    set_seeds(22)
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv, d=1, t=10)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_targeted_attack(bn_targeted_adversarial):
    set_seeds(22)
    adv = bn_targeted_adversarial
    attack = Attack()
    attack(adv, d=1)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf
