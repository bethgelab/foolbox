import numpy as np
import pytest

from foolbox.attacks import BoundaryAttack


def test_attack(bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, iterations=200, verbose=True)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_non_verbose(bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, iterations=200, verbose=False)
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
    attack2 = BoundaryAttack()
    attack2(adv, iterations=200, verbose=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf
    assert adv.distance.value < d1
"""


def test_attack_parameters(bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttack(bn_model, bn_criterion)
    np.random.seed(2)
    starting_point = np.random.uniform(0, 1, size=bn_images[0].shape).astype(
        bn_images.dtype
    )
    advs = attack(
        bn_images,
        bn_labels,
        unpack=False,
        iterations=200,
        starting_point=starting_point,
        log_every_n_steps=2,
        tune_batch_size=False,
        threaded_rnd=False,
        threaded_gen=False,
        alternative_generator=True,
        verbose=True,
    )

    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_parameters2(bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttack(bn_model, bn_criterion)
    advs = attack(
        bn_images,
        bn_labels,
        unpack=False,
        iterations=200,
        alternative_generator=True,
        verbose=True,
    )
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


@pytest.mark.filterwarnings("ignore:Batch size tuning after so few steps")
def test_attack_parameters3(bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttack(bn_model, bn_criterion)
    np.random.seed(2)
    starting_point = np.random.uniform(0, 1, size=bn_images[0].shape).astype(
        bn_images.dtype
    )
    advs = attack(
        bn_images,
        bn_labels,
        unpack=False,
        iterations=200,
        starting_point=starting_point,
        log_every_n_steps=2,
        tune_batch_size=30,
        threaded_rnd=False,
        threaded_gen=False,
        verbose=True,
    )

    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, iterations=200, verbose=True)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_impossible(bn_model, bn_impossible_criterion, bn_images, bn_labels):
    attack = BoundaryAttack(bn_model, bn_impossible_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, iterations=200, verbose=True)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf


@pytest.mark.filterwarnings("ignore:Internal inconsistency, probably caused")
def test_attack_convergence(bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, verbose=True)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf

    attack2 = BoundaryAttack(bn_model, bn_criterion)
    bn_images2 = np.array([adv.perturbed for adv in advs])
    advs2 = attack2(bn_images2, bn_labels, unpack=False, iterations=5000)
    for adv in advs2:
        # should converge
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf
