import numpy as np

from foolbox.batch_attacks import BoundaryAttackPlusPlus
from foolbox.distances import Linf


def test_attack(bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttackPlusPlus(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False,
                  iterations=20, verbose=True)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_linf(bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttackPlusPlus(bn_model, bn_criterion, distance=Linf)
    advs = attack(bn_images, bn_labels, unpack=False,
                  iterations=20, verbose=True)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_non_verbose(bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttackPlusPlus(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False,
                  iterations=20, verbose=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


# TODO: Add this test again
"""
def test_attack_continue(bn_adversarial):
    adv = bn_adversarial
    attack1 = BlendedUniformNoiseAttack()
    attack1(adv)
    d1 = adv.distance.value
    attack2 = BoundaryAttackPlusPlus()
    attack2(adv, iterations=20, verbose=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf
    assert adv.distance.value < d1
"""


def test_attack_targeted(bn_model, bn_criterion, bn_images, bn_labels):
    np.random.seed(2)
    starting_point = np.random.uniform(
        0, 1, size=bn_images[0].shape).astype(bn_images.dtype)

    attack = BoundaryAttackPlusPlus(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False,
                  iterations=21,
                  starting_point=starting_point,
                  log_every_n_steps=2,
                  gamma=0.01,
                  stepsize_search='geometric_progression',
                  batch_size=128,
                  initial_num_evals=200,
                  max_num_evals=20000,
                  verbose=True)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_linf_targeted(bn_model, bn_criterion, bn_images, bn_labels):
    np.random.seed(2)
    starting_point = np.random.uniform(
        0, 1, size=bn_images[0].shape).astype(bn_images.dtype)

    attack = BoundaryAttackPlusPlus(bn_model, bn_criterion, distance=Linf)
    advs = attack(bn_images, bn_labels, unpack=False,
                  iterations=21,
                  starting_point=starting_point,
                  log_every_n_steps=2,
                  gamma=0.01,
                  stepsize_search='grid_search',
                  batch_size=128,
                  initial_num_evals=200,
                  max_num_evals=20000,
                  verbose=True)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttackPlusPlus(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False,
                  iterations=20, verbose=True)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_impossible(bn_model, bn_impossible_criterion, bn_images,
                           bn_labels):
    attack = BoundaryAttackPlusPlus(bn_model, bn_impossible_criterion)
    advs = attack(bn_images, bn_labels, unpack=False,
                  iterations=20, verbose=True)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf
