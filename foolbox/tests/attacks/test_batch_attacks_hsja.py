import numpy as np

from foolbox.attacks import HopSkipJumpAttack, BoundaryAttackPlusPlus
from foolbox.attacks import BlendedUniformNoiseAttack
from foolbox.distances import Linf


def test_attack(bn_model, bn_criterion, bn_images, bn_labels):
    attack = HopSkipJumpAttack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, iterations=20, verbose=True, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_stepsize_gridsearch(bn_model, bn_criterion, bn_images, bn_labels):
    attack = HopSkipJumpAttack(bn_model, bn_criterion)
    advs = attack(
        bn_images,
        bn_labels,
        unpack=False,
        iterations=20,
        verbose=True,
        stepsize_search="grid_search",
    )
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_linf(bn_model, bn_criterion, bn_images, bn_labels):
    attack = HopSkipJumpAttack(bn_model, bn_criterion, distance=Linf)
    advs = attack(bn_images, bn_labels, iterations=20, verbose=True, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_non_verbose(bn_model, bn_criterion, bn_images, bn_labels):
    attack = HopSkipJumpAttack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, iterations=20, verbose=False, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_continue(bn_model, bn_criterion, bn_images, bn_labels):
    attack1 = BlendedUniformNoiseAttack(bn_model, bn_criterion)
    advs1 = attack1(bn_images, bn_labels, unpack=False)

    attack = HopSkipJumpAttack(bn_model, bn_criterion)
    advs2 = attack(
        bn_images,
        bn_labels,
        iterations=21,
        log_every_n_steps=2,
        gamma=0.01,
        stepsize_search="geometric_progression",
        batch_size=128,
        initial_num_evals=200,
        max_num_evals=20000,
        verbose=True,
        individual_kwargs=[{"starting_point": a.perturbed} for a in advs1],
        unpack=False,
    )
    for adv1, adv2 in zip(advs1, advs2):
        assert adv2.perturbed is not None
        assert adv2.distance.value < np.inf
        assert adv2.distance.value < adv1.distance.value


def test_attack_targeted(bn_model, bn_targeted_criterion, bn_images, bn_labels):
    np.random.seed(2)
    starting_point = np.random.uniform(0, 1, size=bn_images[0].shape).astype(
        bn_images.dtype
    )

    attack = HopSkipJumpAttack(bn_model, bn_targeted_criterion)
    advs = attack(
        bn_images,
        bn_labels,
        iterations=21,
        starting_point=starting_point,
        log_every_n_steps=2,
        gamma=0.01,
        stepsize_search="geometric_progression",
        batch_size=128,
        initial_num_evals=200,
        max_num_evals=20000,
        verbose=True,
        unpack=False,
    )
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_linf_targeted(bn_model, bn_targeted_criterion, bn_images, bn_labels):
    np.random.seed(2)
    starting_point = np.random.uniform(0, 1, size=bn_images[0].shape).astype(
        bn_images.dtype
    )

    attack = HopSkipJumpAttack(bn_model, bn_targeted_criterion, distance=Linf)
    advs = attack(
        bn_images,
        bn_labels,
        iterations=21,
        starting_point=starting_point,
        log_every_n_steps=2,
        gamma=0.01,
        stepsize_search="geometric_progression",
        batch_size=128,
        initial_num_evals=200,
        max_num_evals=20000,
        verbose=True,
        unpack=False,
    )
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = HopSkipJumpAttack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, iterations=20, verbose=False, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_impossible(bn_model, bn_impossible_criterion, bn_images, bn_labels):
    attack = HopSkipJumpAttack(bn_model, bn_impossible_criterion)
    advs = attack(bn_images, bn_labels, iterations=20, verbose=False, unpack=False)
    for adv in advs:
        assert adv.perturbed is None


def test_attack_oldname(bn_model, bn_criterion, bn_images, bn_labels):
    attack = BoundaryAttackPlusPlus(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, iterations=20, verbose=True, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf
