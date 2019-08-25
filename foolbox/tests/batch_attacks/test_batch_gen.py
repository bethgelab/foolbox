import pytest
import numpy as np

from foolbox.batch_attacks import GenAttack as Attack


def test_untargeted_attack(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    with pytest.raises(AssertionError, match='targeted'):
        attack(bn_images, bn_labels, unpack=False)


def test_attack(bn_model, bn_targeted_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_targeted_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf