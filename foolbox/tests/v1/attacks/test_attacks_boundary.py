import numpy as np
import pytest

from foolbox.v1.attacks import BoundaryAttack
from foolbox.v1.attacks import DeepFoolAttack
from foolbox.v1.attacks import BlendedUniformNoiseAttack


def test_attack(bn_adversarial):
    adv = bn_adversarial
    attack = BoundaryAttack()
    attack(adv, iterations=200, verbose=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_non_verbose(bn_adversarial):
    adv = bn_adversarial
    attack = BoundaryAttack()
    attack(adv, iterations=200, verbose=False)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


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


def test_attack_parameters(bn_adversarial):
    adv = bn_adversarial
    attack = BoundaryAttack()
    o = adv.unperturbed
    np.random.seed(2)
    starting_point = np.random.uniform(0, 1, size=o.shape).astype(o.dtype)
    attack(
        adv,
        iterations=200,
        starting_point=starting_point,
        log_every_n_steps=2,
        tune_batch_size=False,
        threaded_rnd=False,
        threaded_gen=False,
        alternative_generator=True,
        verbose=True,
    )
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_parameters2(bn_adversarial):
    adv = bn_adversarial
    attack = BoundaryAttack()
    attack(adv, iterations=200, alternative_generator=True, verbose=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


@pytest.mark.filterwarnings("ignore:Batch size tuning after so few steps")
def test_attack_parameters3(bn_adversarial):
    adv = bn_adversarial
    attack = BoundaryAttack()
    o = adv.unperturbed
    np.random.seed(2)
    starting_point = np.random.uniform(0, 1, size=o.shape).astype(o.dtype)
    attack(
        adv,
        iterations=200,
        starting_point=starting_point,
        log_every_n_steps=2,
        tune_batch_size=30,
        threaded_rnd=False,
        threaded_gen=False,
        verbose=True,
    )
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = BoundaryAttack()
    attack(adv, iterations=200, verbose=True)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_impossible(bn_impossible):
    adv = bn_impossible
    attack = BoundaryAttack()
    attack(adv, iterations=200, verbose=True)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf


@pytest.mark.filterwarnings("ignore:Internal inconsistency, probably caused")
def test_attack_convergence(bn_adversarial):
    adv = bn_adversarial
    attack1 = DeepFoolAttack()
    attack1(adv)
    attack2 = BoundaryAttack()
    attack2(adv, iterations=5000, verbose=True)
    # should converge
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf
