import pytest
import numpy as np

from foolbox.batch_attacks import LinfinityBasicIterativeAttack
from foolbox.batch_attacks import L1BasicIterativeAttack
from foolbox.batch_attacks import L2BasicIterativeAttack
from foolbox.batch_attacks import ProjectedGradientDescentAttack
from foolbox.batch_attacks import RandomStartProjectedGradientDescentAttack
from foolbox.batch_attacks import MomentumIterativeAttack

from foolbox.distances import Linfinity
from foolbox.distances import MAE

Attacks = [
    LinfinityBasicIterativeAttack,
    L1BasicIterativeAttack,
    L2BasicIterativeAttack,
    ProjectedGradientDescentAttack,
    RandomStartProjectedGradientDescentAttack,
    MomentumIterativeAttack,
]


def test_attack_no_binary_search_and_no_return_early(bn_model, bn_criterion, bn_images, bn_labels):
    attack = LinfinityBasicIterativeAttack(bn_model, bn_criterion, distance=Linfinity)
    advs = attack(bn_images, bn_labels, unpack=False, binary_search=False, return_early=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_linf(Attack, bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, binary_search=10)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_l2(Attack, bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_l1(Attack, bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion, distance=MAE)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_targeted_attack(Attack, bn_model, bn_targeted_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_targeted_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_gl(Attack, gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_impossible(Attack, bn_model, bn_impossible_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_impossible_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf
