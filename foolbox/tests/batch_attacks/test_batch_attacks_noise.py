import pytest
import numpy as np

from foolbox.batch_attacks import AdditiveUniformNoiseAttack
from foolbox.batch_attacks import AdditiveGaussianNoiseAttack
from foolbox.batch_attacks import SaltAndPepperNoiseAttack
from foolbox.batch_attacks import BlendedUniformNoiseAttack

Attacks = [
    AdditiveUniformNoiseAttack,
    AdditiveGaussianNoiseAttack,
    SaltAndPepperNoiseAttack,
    BlendedUniformNoiseAttack,
]


@pytest.mark.parametrize('Attack', Attacks)
def test_attack(Attack, bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_gl(Attack, gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


@pytest.mark.parametrize('Attack', Attacks)
def test_attack_impossible(Attack, bn_model, bn_impossible_criterion, bn_images,
                           bn_labels):
    attack = Attack(bn_model, bn_impossible_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf
