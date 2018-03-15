import numpy as np

from foolbox.attacks import ResetAttack as Attack


def test_attack(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_attack_startingpoint(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    o = adv.original_image
    np.random.seed(2)
    starting_point = np.random.uniform(
        0, 1, size=o.shape).astype(o.dtype)
    attack(adv, starting_point=starting_point)
    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_attack_continue(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    o = adv.original_image
    np.random.seed(2)
    starting_point = np.random.uniform(
        0, 1, size=o.shape).astype(o.dtype)
    adv.predictions(starting_point)
    assert adv.image is not None
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_attack_impossible(bn_impossible):
    adv = bn_impossible
    attack = Attack()
    attack(adv)
    assert adv.image is None
    assert adv.distance.value == np.inf
