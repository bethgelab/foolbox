import numpy as np

from foolbox.v1.attacks import CarliniWagnerL2Attack as Attack


def test_untargeted_attack(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, max_iterations=100)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_targeted_attack(bn_targeted_adversarial):
    adv = bn_targeted_adversarial
    attack = Attack()
    attack(adv, max_iterations=100, binary_search_steps=20)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_impossible(bn_impossible):
    adv = bn_impossible
    attack = Attack()
    attack(adv, max_iterations=100)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf


def test_attack_gl(gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv, max_iterations=100)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf
