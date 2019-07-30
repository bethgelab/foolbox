import numpy as np
import pytest

from foolbox.batch_attacks import GradientAttack as Attack

from foolbox import run_parallel
from foolbox import run_sequential
from foolbox.distances import MSE


def test_run_parallel(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs1 = attack(bn_images, bn_labels)

    advs2 = run_parallel(Attack, bn_model, bn_criterion, bn_images, bn_labels)
    advs2 = np.stack([a.perturbed for a in advs2])

    assert np.all(advs1 == advs2)


def test_run_parallel_invalid_arguments(bn_model, bn_criterion, bn_images,
                                        bn_labels):
    labels_wrong = bn_labels[[0]]
    criteria_wrong = [bn_criterion] * (len(bn_images) + 1)
    distances_wrong = [MSE] * (len(bn_images) + 1)

    # test too few labels
    with pytest.raises(AssertionError):
        run_parallel(Attack, bn_model, bn_criterion, bn_images,
                     labels_wrong)

    # test too many criteria
    with pytest.raises(AssertionError):
        run_parallel(Attack, bn_model, criteria_wrong, bn_images,
                     bn_labels)

    # test too many distances
    with pytest.raises(AssertionError):
        run_parallel(Attack, bn_model, bn_criterion, bn_images,
                     bn_labels, distance=distances_wrong)


def test_run_sequential(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs1 = attack(bn_images, bn_labels)

    advs2 = run_sequential(Attack, bn_model, bn_criterion, bn_images, bn_labels)
    advs2 = np.stack([a.perturbed for a in advs2])

    assert np.all(advs1 == advs2)


def test_run_sequential_invalid_arguments(bn_model, bn_criterion, bn_images,
                                          bn_labels):
    labels_wrong = bn_labels[[0]]
    criteria_wrong = [bn_criterion] * (len(bn_images) + 1)
    distances_wrong = [MSE] * (len(bn_images) + 1)

    # test too few labels
    with pytest.raises(AssertionError):
        run_sequential(Attack, bn_model, bn_criterion, bn_images,
                       labels_wrong)

    # test too many criteria
    with pytest.raises(AssertionError):
        run_sequential(Attack, bn_model, criteria_wrong, bn_images,
                       bn_labels)

    # test too many distances
    with pytest.raises(AssertionError):
        run_sequential(Attack, bn_model, bn_criterion, bn_images,
                       bn_labels, distance=distances_wrong)
