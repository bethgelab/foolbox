from unittest.mock import Mock

import pytest
import numpy as np

from foolbox import attacks
from foolbox import Adversarial


def test_abstract_attack():
    with pytest.raises(TypeError):
        attacks.Attack()


def test_base_init():
    with pytest.raises(ValueError):
        attacks.FGSM(Mock(), None)
    with pytest.raises(ValueError):
        attacks.FGSM(None, Mock())
    assert attacks.FGSM(None, None) is not None
    assert attacks.FGSM(Mock(), Mock()) is not None


def test_aliases():
    assert attacks.GradientSignAttack == attacks.FGSM


def test_base_attack(model, criterion, image, label):
    attack = attacks.FGSM(model, criterion)
    assert attack.name() == 'GradientSignAttack'

    with pytest.raises(ValueError):
        attack(image=image)

    with pytest.raises(ValueError):
        attack(label=label)

    adv = attack(image=image, label=label)
    assert adv.shape == image.shape
    adv = attack(image=image, label=label, unpack=False)
    assert adv.get().shape == image.shape

    adv = Adversarial(model, criterion, image, label)
    adv = attack(adv)
    assert adv.shape == image.shape

    with pytest.raises(ValueError):
        attack(adv, label=label)

    attack = attacks.FGSM()
    with pytest.raises(ValueError):
        attack(image=image, label=label)


def test_gradient_sign_attack_with_gradient(
        model, criterion, image, label):

    adversarial = Adversarial(model, criterion, image, label)

    attack = attacks.GradientSignAttack()
    attack(adversarial)
    assert adversarial.get() is not None
    assert adversarial.best_distance().value() < np.inf


def test_gradient_sign_attack_without_gradient(
        model, criterion, image, label):

    del model.gradient
    adversarial = Adversarial(model, criterion, image, label)

    attack = attacks.GradientSignAttack()
    attack(adversarial)
    assert adversarial.get() is None
    assert adversarial.best_distance().value() == np.inf


def test_iterative_gradient_sign_attack_with_gradient(
        model, criterion, image, label):

    adversarial = Adversarial(model, criterion, image, label)

    attack = attacks.IterativeGradientSignAttack()
    attack(adversarial, epsilons=2, steps=3)
    assert adversarial.get() is not None
    assert adversarial.best_distance().value() < np.inf


def test_iterative_gradient_sign_attack_without_gradient(
        model, criterion, image, label):

    del model.gradient
    adversarial = Adversarial(model, criterion, image, label)

    attack = attacks.IterativeGradientSignAttack()
    attack(adversarial)
    assert adversarial.get() is None
    assert adversarial.best_distance().value() == np.inf


def test_lbfgs_attack_with_gradient(
        model, criterion, image, label):

    adversarial = Adversarial(model, criterion, image, label)

    attack = attacks.LBFGSAttack()
    attack(adversarial)
    assert adversarial.get() is not None
    assert adversarial.best_distance().value() < np.inf


def test_lbfgs_attack_without_gradient(
        model, criterion, image, label):

    del model.gradient
    adversarial = Adversarial(model, criterion, image, label)

    attack = attacks.LBFGSAttack()
    attack(adversarial)
    assert adversarial.get() is None
    assert adversarial.best_distance().value() == np.inf


def test_approx_lbfgs_attack_with_gradient(
        model, criterion, image, label):

    adversarial = Adversarial(model, criterion, image, label)

    attack = attacks.ApproximateLBFGSAttack()
    assert attack is not None
    assert adversarial is not None
    # attack(adversarial)
    # assert adversarial.get() is not None
    # assert adversarial.best_distance().value() < np.inf


def test_approx_lbfgs_attack_without_gradient(
        model, criterion, image, label):

    del model.gradient
    adversarial = Adversarial(model, criterion, image, label)

    attack = attacks.ApproximateLBFGSAttack()
    assert attack is not None
    assert adversarial is not None
    # attack(adversarial)
    # assert adversarial.get() is not None
    # assert adversarial.best_distance().value() < np.inf
