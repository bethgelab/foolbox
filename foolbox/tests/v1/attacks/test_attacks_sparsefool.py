import numpy as np

from foolbox.v1.attacks import SparseFoolAttack as Attack


def test_attack(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf


def test_targeted_attack(bn_targeted_adversarial):
    adv = bn_targeted_adversarial
    attack = Attack()
    attack(adv)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf


def test_lambda(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, lambda_=1.5)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_subsample(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, subsample=5)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_step(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, steps=1)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_impossible(bn_impossible):
    adv = bn_impossible
    attack = Attack()
    attack(adv)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf
