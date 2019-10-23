import numpy as np

from foolbox.v1.attacks import ApproximateLBFGSAttack as Attack


def test_name():
    attack = Attack()
    assert "Approx" in attack.name()


def test_attack(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, maxiter=1, epsilon=1000)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv, maxiter=1, epsilon=1000)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


# def test_targeted_attack(bn_targeted_adversarial):
#     adv = bn_targeted_adversarial
#     attack = Attack()
#     attack(adv)
#     assert adv.perturbed is not None
#     assert adv.distance.value < np.inf
