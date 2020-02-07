import numpy as np
import pytest

from foolbox.attacks import PrecomputedAdversarialsAttack as Attack


def test_attack(bn_model, bn_criterion, bn_images, bn_labels):
    input_images = bn_images
    output_images = np.zeros_like(input_images)
    attack = Attack(bn_model, bn_criterion)
    advs = attack(
        bn_images,
        bn_labels,
        unpack=False,
        candidate_inputs=input_images,
        candidate_outputs=output_images,
    )
    for adv in advs:
        assert adv.perturbed is not None
        assert adv.distance.value < np.inf


def test_unknown_image(bn_model, bn_criterion, bn_images, bn_labels):
    input_images = np.zeros_like(bn_images)
    output_images = np.zeros_like(input_images)
    attack = Attack(bn_model, bn_criterion)

    with pytest.raises(ValueError):
        attack(
            bn_images,
            bn_labels,
            unpack=False,
            candidate_inputs=input_images,
            candidate_outputs=output_images,
        )
