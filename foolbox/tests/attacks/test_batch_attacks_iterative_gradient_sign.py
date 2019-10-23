import numpy as np

from foolbox.attacks import IterativeGradientSignAttack as Attack


def test_attack(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, epsilons=10, steps=5)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, epsilons=10, steps=5)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf
