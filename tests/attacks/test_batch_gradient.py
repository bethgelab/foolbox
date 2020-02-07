import pytest
import numpy as np

from foolbox.gradient_estimators import CoordinateWiseGradientEstimator
from foolbox.gradient_estimators import EvolutionaryStrategiesGradientEstimator

from foolbox.models import ModelWithEstimatedGradients

from foolbox.attacks import GradientAttack as Attack


def test_untargeted_attack(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_eps(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs = attack(
        bn_images, bn_labels, unpack=False, epsilons=np.linspace(0.0, 1.0, 100)[1:]
    )
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(gl_bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf


@pytest.fixture(
    params=[CoordinateWiseGradientEstimator, EvolutionaryStrategiesGradientEstimator]
)
def test_attack_eg(request, bn_model, bn_criterion, bn_images, bn_labels):
    GradientEstimator = request.param
    gradient_estimator = GradientEstimator(epsilon=0.01)
    model = ModelWithEstimatedGradients(bn_model, gradient_estimator)
    attack = Attack(model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf
