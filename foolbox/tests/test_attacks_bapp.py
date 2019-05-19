import numpy as np

from foolbox.attacks import BoundaryAttackPlusPlus
from foolbox.attacks import BlendedUniformNoiseAttack
from foolbox.distances import Linf


def test_attack(bn_adversarial):
    adv = bn_adversarial
    attack = BoundaryAttackPlusPlus()
    attack(adv, iterations=20, verbose=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_linf(bn_adversarial):
    adv = bn_adversarial
    attack = BoundaryAttackPlusPlus(distance=Linf)
    attack(adv, iterations=20, verbose=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_non_verbose(bn_adversarial):
    adv = bn_adversarial
    attack = BoundaryAttackPlusPlus()
    attack(adv, iterations=20, verbose=False)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


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


def test_attack_targeted(bn_adversarial):
    adv = bn_adversarial
    attack = BoundaryAttackPlusPlus()
    o = adv.unperturbed
    np.random.seed(2)
    starting_point = np.random.uniform(
        0, 1, size=o.shape).astype(o.dtype)
    attack(
        adv,
        iterations=21,
        starting_point=starting_point,
        log_every_n_steps=2,
        gamma=0.01,
        stepsize_search='geometric_progression',
        batch_size=128,
        initial_num_evals=200,
        max_num_evals=20000,
        verbose=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_linf_targeted(bn_adversarial):
    adv = bn_adversarial
    attack = BoundaryAttackPlusPlus(distance=Linf)
    o = adv.unperturbed
    np.random.seed(2)
    starting_point = np.random.uniform(
        0, 1, size=o.shape).astype(o.dtype)
    attack(
        adv,
        iterations=21,
        starting_point=starting_point,
        log_every_n_steps=2,
        gamma=0.01,
        stepsize_search='grid_search',
        batch_size=128,
        initial_num_evals=200,
        max_num_evals=20000,
        verbose=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = BoundaryAttackPlusPlus()
    attack(adv, iterations=200, verbose=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_impossible(bn_impossible):
    adv = bn_impossible
    attack = BoundaryAttackPlusPlus()
    attack(adv, iterations=200, verbose=True)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf
