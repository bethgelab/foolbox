import numpy as np

from foolbox.attacks import PointwiseAttack as Attack


def test_attack(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_startingpoint(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    np.random.seed(2)
    starting_point = np.random.uniform(0, 1, size=bn_images[0].shape).astype(
        bn_images.dtype
    )
    advs = attack(bn_images, bn_labels, unpack=False, starting_point=starting_point)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


# TODO: Add this test again
"""
def test_attack_continue(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    o = adv.unperturbed
    np.random.seed(2)
    starting_point = np.random.uniform(
        0, 1, size=o.shape).astype(o.dtype)
    adv.forward_one(starting_point)
    assert adv.perturbed is not None
    attack(adv)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf
"""


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_impossible(bn_model, bn_impossible_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_impossible_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf
