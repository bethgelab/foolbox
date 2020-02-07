import numpy as np

from foolbox.attacks import GradientSignAttack as Attack


def test_untargeted_attack(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_eps(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(
        bn_images, bn_labels, unpack=False, epsilons=np.linspace(0.0, 1.0, 100)[1:]
    )
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf
