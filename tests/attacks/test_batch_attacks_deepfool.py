import pytest
import numpy as np

from foolbox.attacks import DeepFoolAttack
from foolbox.attacks import DeepFoolL2Attack
from foolbox.attacks import DeepFoolLinfinityAttack

from foolbox.distances import MAE
from foolbox.distances import Linf

Attacks = [DeepFoolAttack, DeepFoolL2Attack, DeepFoolLinfinityAttack]


@pytest.mark.parametrize("Attack", Attacks)
def test_attack(Attack, bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


@pytest.mark.parametrize("Attack", Attacks)
def test_attack_gl(Attack, gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf


@pytest.mark.parametrize("Attack", Attacks)
def test_targeted_attack(Attack, bn_model, bn_targeted_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_targeted_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf


@pytest.mark.parametrize("Attack", Attacks)
def test_subsample(Attack, bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False, subsample=5)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


@pytest.mark.parametrize("Attack", Attacks)
def test_attack_impossible(
    Attack, bn_model, bn_impossible_criterion, bn_images, bn_labels
):
    attack = Attack(bn_model, bn_impossible_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf


def test_deepfool_auto_linf(bn_model, bn_criterion, bn_images, bn_labels):
    attack = DeepFoolAttack(bn_model, bn_criterion, distance=Linf)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_deepfool_auto_mae(bn_model, bn_criterion, bn_images, bn_labels):
    attack = DeepFoolAttack(bn_model, bn_criterion, distance=MAE)
    with pytest.raises(NotImplementedError):
        attack(bn_images, bn_labels, unpack=False)


def test_deepfool_auto_p0(bn_model, bn_criterion, bn_images, bn_labels):
    attack = DeepFoolAttack(bn_model, bn_criterion)
    with pytest.raises(ValueError):
        attack(bn_images, bn_labels, unpack=False, p=0)
