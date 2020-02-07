import numpy as np

from foolbox import set_seeds
from foolbox.attacks import LocalSearchAttack as Attack


def test_attack(bn_model, bn_criterion, bn_images, bn_labels):
    set_seeds(22)
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, d=1, t=20, R=250)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    set_seeds(22)
    attack = Attack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, d=1, t=20, R=250)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_targeted_attack(bn_model, bn_targeted_criterion, bn_images, bn_labels):
    set_seeds(22)
    attack = Attack(bn_model, bn_targeted_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, d=1, t=20, R=250)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf
