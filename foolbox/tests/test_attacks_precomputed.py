import numpy as np
import pytest

from foolbox.attacks import PrecomputedImagesAttack as Attack


def test_attack(bn_adversarial):
    adv = bn_adversarial

    image = adv.original_image
    input_images = image[np.newaxis]
    output_images = np.zeros_like(input_images)

    attack = Attack(input_images, output_images)

    attack(adv)

    assert adv.image is not None
    assert adv.distance.value < np.inf


def test_unknown_image(bn_adversarial):
    adv = bn_adversarial

    image = adv.original_image
    input_images = np.zeros_like(image[np.newaxis])
    output_images = np.zeros_like(input_images)

    attack = Attack(input_images, output_images)

    with pytest.raises(ValueError):
        attack(adv)

    assert adv.image is None
    assert adv.distance.value == np.inf
