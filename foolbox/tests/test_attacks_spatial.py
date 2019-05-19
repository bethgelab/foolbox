import numpy as np

from foolbox.attacks import SpatialAttack as Attack


def test_attack_pytorch(bn_adversarial_pytorch):
    adv = bn_adversarial_pytorch
    attack = Attack()
    attack(adv)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_rnd(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, random_sampling=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_norot(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, do_rotations=False)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_notrans(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, do_translations=False)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_notrans_norot(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, do_translations=False, do_rotations=False)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf


def test_attack_gl(gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf
