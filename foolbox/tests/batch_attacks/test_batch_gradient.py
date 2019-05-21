import pytest
import numpy as np
from contextlib import contextmanager

from foolbox.gradient_estimators import CoordinateWiseGradientEstimator
from foolbox.gradient_estimators import EvolutionaryStrategiesGradientEstimator

from foolbox.models import ModelWithEstimatedGradients

from foolbox.batch_attacks import GradientAttack as Attack


def test_untargeted_attack(bn_model, bn_criterion, bn_images, bn_labels):
    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        attack = Attack(model, bn_criterion)
        advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_eps(bn_model, bn_criterion, bn_images, bn_labels):
    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        attack = Attack(model, bn_criterion)
        advs = attack(bn_images, bn_labels, unpack=False, epsilons=np.linspace(0., 1., 100)[1:])
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    cm_model = contextmanager(gl_bn_model)
    with cm_model() as model:
        attack = Attack(model, bn_criterion)
        advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf


def eg_bn_model_factory(request, bn_model):
    GradientEstimator = request.param

    def eg_bn_model():
        cm_model = contextmanager(bn_model)
        with cm_model() as model:
            gradient_estimator = GradientEstimator(epsilon=0.01)
            model = ModelWithEstimatedGradients(model, gradient_estimator)
            yield model
    return eg_bn_model


@pytest.fixture(params=[CoordinateWiseGradientEstimator,
                        EvolutionaryStrategiesGradientEstimator])
def test_attack_eg(request, bn_model, bn_criterion, bn_images, bn_labels):
    eg_bn_model = eg_bn_model_factory(request, bn_model)
    cm_model = contextmanager(eg_bn_model)
    with cm_model() as model:
        attack = Attack(model, bn_criterion)
        advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf
