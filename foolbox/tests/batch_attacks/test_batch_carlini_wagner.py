import numpy as np
from contextlib import contextmanager

from foolbox.batch_attacks import CarliniWagnerL2Attack as Attack


def test_untargeted_attack(bn_model, bn_criterion, bn_images, bn_labels):
    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        attack = Attack(model, bn_criterion)
        advs = attack(bn_images, bn_labels, unpack=False, max_iterations=100)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_targeted_attack(bn_model, bn_targeted_criterion, bn_images, bn_labels):
    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        attack = Attack(model, bn_targeted_criterion)
        advs = attack(bn_images, bn_labels, unpack=False, max_iterations=100, binary_search_steps=20)
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_attack_impossible(bn_model, bn_impossible_criterion, bn_images, bn_labels):
    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        attack = Attack(model, bn_impossible_criterion)
        advs = attack(bn_images, bn_labels, unpack=False, max_iterations=100, binary_search_steps=20)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf


def test_attack_gl(gl_bn_model, bn_criterion, bn_images, bn_labels):
    cm_model = contextmanager(gl_bn_model)
    with cm_model() as model:
        attack = Attack(model, bn_criterion)
        advs = attack(bn_images, bn_labels, unpack=False, max_iterations=100)
    for adv in advs:
        assert adv.perturbed is None
        assert adv.distance.value == np.inf
