import pytest
import numpy as np

from foolbox.batch_attacks import GradientAttack
from foolbox.batch_attacks import BinarizationRefinementAttack


def test_attack(bn_model, bn_criterion, bn_images, binarized_bn_labels):
    attack = GradientAttack(bn_model, bn_criterion)
    advs = attack(bn_images, binarized_bn_labels, unpack=False)

    for adv in advs:
        assert adv.perturbed is not None

    attack = BinarizationRefinementAttack(bn_model, bn_criterion)
    advs2 = attack(
        bn_images, binarized_bn_labels, unpack=False,
        individual_kwargs=[{'starting_point': adv.perturbed} for adv in advs]
    )

    for adv1, adv2 in zip(advs, advs2):
        v1 = adv1.distance.value
        v2 = adv2.distance.value

        assert v2 < v1 < np.inf

        o = adv2.unperturbed
        x = adv2.perturbed
        d = x[x != o]
        np.testing.assert_allclose(d, 0.5)


def test_attack_fail(bn_model, bn_criterion, bn_images, bn_labels):
    attack = GradientAttack(bn_model, bn_criterion)
    advs = attack(bn_images, bn_labels, unpack=False)
    for adv in advs:
        assert adv is not None

    attack = BinarizationRefinementAttack(bn_model, bn_criterion)
    with pytest.raises(AssertionError) as e:
        attack(
            bn_images, bn_labels,
            individual_kwargs=[{'starting_point': adv.perturbed} for adv in advs])
    assert 'threshold does not match' in str(e.value)


def test_attack_noinit(bn_model, bn_criterion, bn_images, binarized_bn_labels):
    attack = BinarizationRefinementAttack(bn_model, bn_criterion)
    advs = attack(
        bn_images, binarized_bn_labels, unpack=False
    )
    for adv in advs:
        assert adv.perturbed is None


def test_attack2(bn_model, bn_criterion, bn_images, binarized2_bn_labels):
    attack = GradientAttack(bn_model, bn_criterion)
    advs = attack(bn_images, binarized2_bn_labels, unpack=False)

    attack = BinarizationRefinementAttack(bn_model, bn_criterion)
    advs2 = attack(
        bn_images, binarized2_bn_labels, unpack=False,
        individual_kwargs=[{'starting_point': adv.perturbed} for adv in advs]
    )

    for adv1, adv2 in zip(advs, advs2):
        v1 = adv1.distance.value
        v2 = adv2.distance.value

        assert v2 < v1 < np.inf

        o = adv2.unperturbed
        x = adv2.perturbed
        d = x[x != o]
        np.testing.assert_allclose(d, 0.5)


def test_attack_wrong_arg(bn_model, bn_criterion, bn_images, binarized2_bn_labels):
    attack = GradientAttack(bn_model, bn_criterion)
    advs = attack(bn_images, binarized2_bn_labels, unpack=False)

    attack = BinarizationRefinementAttack(bn_model, bn_criterion)
    with pytest.raises(ValueError):
        attack(
            bn_images, binarized2_bn_labels, unpack=False,
            individual_kwargs=[{'starting_point': adv.perturbed} for adv in advs],
            included_in='blabla'
        )
