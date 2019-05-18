import numpy as np

from foolbox.attacks import SaliencyMapAttack as Attack


def test_attack(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_random_targets(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, num_random_targets=2)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_targeted_attack(bn_targeted_adversarial):
    adv = bn_targeted_adversarial
    attack = Attack()
    attack(adv)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_targeted_attack_slow(bn_targeted_adversarial):
    adv = bn_targeted_adversarial
    attack = Attack()
    attack(adv, fast=False)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_targeted_attack_max(bn_targeted_adversarial):
    adv = bn_targeted_adversarial
    attack = Attack()
    attack(adv, max_perturbations_per_pixel=1)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf
