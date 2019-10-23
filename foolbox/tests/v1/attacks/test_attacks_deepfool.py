import pytest
import numpy as np

from foolbox.v1.attacks import DeepFoolAttack
from foolbox.v1.attacks import DeepFoolL2Attack
from foolbox.v1.attacks import DeepFoolLinfinityAttack

Attacks = [DeepFoolAttack, DeepFoolL2Attack, DeepFoolLinfinityAttack]


@pytest.mark.parametrize("Attack", Attacks)
def test_attack(Attack, bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


@pytest.mark.parametrize("Attack", Attacks)
def test_attack_gl(Attack, gl_bn_adversarial):
    adv = gl_bn_adversarial
    attack = Attack()
    attack(adv)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf


@pytest.mark.parametrize("Attack", Attacks)
def test_targeted_attack(Attack, bn_targeted_adversarial):
    adv = bn_targeted_adversarial
    attack = Attack()
    attack(adv)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf


@pytest.mark.parametrize("Attack", Attacks)
def test_subsample(Attack, bn_adversarial):
    adv = bn_adversarial
    attack = Attack()
    attack(adv, subsample=5)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


@pytest.mark.parametrize("Attack", Attacks)
def test_attack_impossible(Attack, bn_impossible):
    adv = bn_impossible
    attack = Attack()
    attack(adv)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf


def test_deepfool_auto_linf(bn_adversarial_linf):
    adv = bn_adversarial_linf
    attack = DeepFoolAttack()
    attack(adv)
    assert adv.perturbed is not None
    assert adv.distance.value < np.inf


def test_deepfool_auto_mae(bn_adversarial_mae):
    adv = bn_adversarial_mae
    attack = DeepFoolAttack()
    with pytest.raises(NotImplementedError):
        attack(adv)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf


def test_deepfool_auto_p0(bn_adversarial):
    adv = bn_adversarial
    attack = DeepFoolAttack()
    with pytest.raises(ValueError):
        attack(adv, p=0)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf
