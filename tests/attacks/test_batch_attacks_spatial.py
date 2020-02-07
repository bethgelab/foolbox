import numpy as np

from foolbox.attacks import SpatialAttack as Attack


def test_attack(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_rnd(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, random_sampling=True)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_norot(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, do_rotations=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_notrans(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, do_translations=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_notrans_norot(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(
        bn_images, bn_labels, unpack=False, do_translations=False, do_rotations=False
    )
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf
