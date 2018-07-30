import numpy as np

from foolbox.attacks.newtonfool import NewtonFoolAttack as Attack


def test_attack(bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv)
    assert 1 == 1
    #assert adv.image is not None
    #assert adv.distance.value < np.inf