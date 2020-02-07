import numpy as np
import pytest

from foolbox.attacks import GradientAttack as Attack

from foolbox import run_parallel
from foolbox import run_sequential
from foolbox.distances import MSE


def test_run_parallel(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs1 = attack(bn_images, bn_labels)

    advs2 = run_parallel(Attack, bn_model, bn_criterion, bn_images, bn_labels)
    advs2 = np.stack([a.perturbed for a in advs2])

    assert np.all(advs1 == advs2)


def test_run_multiple_kwargs(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs1 = attack(
        bn_images,
        bn_labels,
        unpack=False,
        individual_kwargs=[{"epsilons": 900} for _ in bn_images],
    )
    advs1p = run_parallel(
        Attack,
        bn_model,
        bn_criterion,
        bn_images,
        bn_labels,
        individual_kwargs=[{"epsilons": 900} for _ in bn_images],
    )
    advs1s = run_sequential(
        Attack,
        bn_model,
        bn_criterion,
        bn_images,
        bn_labels,
        individual_kwargs=[{"epsilons": 900} for _ in bn_images],
    )

    advs2 = attack(
        bn_images,
        bn_labels,
        unpack=False,
        individual_kwargs=[{"epsilons": 900} for _ in bn_images],
        max_epsilon=0.99,
    )
    advs2p = run_parallel(
        Attack,
        bn_model,
        bn_criterion,
        bn_images,
        bn_labels,
        individual_kwargs=[{"epsilons": 900} for _ in bn_images],
        max_epsilon=0.99,
    )
    advs2s = run_sequential(
        Attack,
        bn_model,
        bn_criterion,
        bn_images,
        bn_labels,
        individual_kwargs=[{"epsilons": 900} for _ in bn_images],
        max_epsilon=0.99,
    )

    advs3 = attack(bn_images, bn_labels, unpack=False, max_epsilon=0.99)
    advs3p = run_parallel(
        Attack, bn_model, bn_criterion, bn_images, bn_labels, max_epsilon=0.99
    )
    advs3s = run_sequential(
        Attack, bn_model, bn_criterion, bn_images, bn_labels, max_epsilon=0.99
    )

    for i in range(len(bn_images)):
        assert advs1[i].perturbed is not None
        assert advs2[i].perturbed is not None
        assert advs3[i].perturbed is not None

        assert advs1p[i].perturbed is not None
        assert advs2p[i].perturbed is not None
        assert advs3p[i].perturbed is not None

        assert advs1s[i].perturbed is not None
        assert advs2s[i].perturbed is not None
        assert advs3s[i].perturbed is not None

        assert np.allclose(advs1[i].perturbed, advs1p[i].perturbed)
        assert np.allclose(advs1[i].perturbed, advs1s[i].perturbed)

        assert np.allclose(advs2[i].perturbed, advs2p[i].perturbed)
        assert np.allclose(advs2[i].perturbed, advs2s[i].perturbed)

        assert np.allclose(advs3[i].perturbed, advs3p[i].perturbed)
        assert np.allclose(advs3[i].perturbed, advs3s[i].perturbed)


def test_run_parallel_invalid_arguments(bn_model, bn_criterion, bn_images, bn_labels):
    labels_wrong = bn_labels[[0]]
    criteria_wrong = [bn_criterion] * (len(bn_images) + 1)
    distances_wrong = [MSE] * (len(bn_images) + 1)
    individual_kwargs_wrong = [{"max_epsilon": 1}] * (len(bn_images) + 1)

    # test too few labels
    with pytest.raises(AssertionError):
        run_parallel(Attack, bn_model, bn_criterion, bn_images, labels_wrong)

    # test too many criteria
    with pytest.raises(AssertionError):
        run_parallel(Attack, bn_model, criteria_wrong, bn_images, bn_labels)

    # test too many distances
    with pytest.raises(AssertionError):
        run_parallel(
            Attack,
            bn_model,
            bn_criterion,
            bn_images,
            bn_labels,
            distance=distances_wrong,
        )

    # test too many kwargs
    with pytest.raises(AssertionError):
        run_parallel(
            Attack,
            bn_model,
            bn_criterion,
            bn_images,
            bn_labels,
            individual_kwargs=individual_kwargs_wrong,
        )


def test_run_sequential(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs1 = attack(bn_images, bn_labels)

    advs2 = run_sequential(Attack, bn_model, bn_criterion, bn_images, bn_labels)
    advs2 = np.stack([a.perturbed for a in advs2])

    assert np.all(advs1 == advs2)


def test_run_sequential_invalid_arguments(bn_model, bn_criterion, bn_images, bn_labels):
    labels_wrong = bn_labels[[0]]
    criteria_wrong = [bn_criterion] * (len(bn_images) + 1)
    distances_wrong = [MSE] * (len(bn_images) + 1)
    individual_kwargs_wrong = [{"max_epsilon": 1}] * (len(bn_images) + 1)

    # test too few labels
    with pytest.raises(AssertionError):
        run_sequential(Attack, bn_model, bn_criterion, bn_images, labels_wrong)

    # test too many criteria
    with pytest.raises(AssertionError):
        run_sequential(Attack, bn_model, criteria_wrong, bn_images, bn_labels)

    # test too many distances
    with pytest.raises(AssertionError):
        run_sequential(
            Attack,
            bn_model,
            bn_criterion,
            bn_images,
            bn_labels,
            distance=distances_wrong,
        )

    # test too many kwargs
    with pytest.raises(AssertionError):
        run_sequential(
            Attack,
            bn_model,
            bn_criterion,
            bn_images,
            bn_labels,
            individual_kwargs=individual_kwargs_wrong,
        )
