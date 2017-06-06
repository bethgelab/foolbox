from unittest.mock import Mock

import numpy as np

from foolbox import Adversarial
from foolbox.distances import MSE


def test_adversarial(model, criterion, image, label):
    adversarial = Adversarial(model, criterion, image, label)

    assert adversarial.get() is None
    assert adversarial.best_distance() == MSE(value=np.inf)
    assert adversarial.original_image() is image
    assert adversarial.original_class() == label
    assert adversarial.target_class() is None
    assert adversarial.normalized_distance(image) == MSE(value=0)
    assert adversarial.normalized_distance(image).value() == 0

    predictions, is_adversarial = adversarial.predictions(image)
    first_predictions = predictions
    assert is_adversarial

    predictions, is_adversarial = adversarial.batch_predictions(image[np.newaxis])  # noqa: E501
    assert (predictions == first_predictions[np.newaxis]).all()
    assert np.all(is_adversarial == np.array([True]))

    predictions, is_adversarial, index = adversarial.batch_predictions(image[np.newaxis], increasing=True)  # noqa: E501
    assert (predictions == first_predictions[np.newaxis]).all()
    assert is_adversarial
    assert index == 0

    predictions, gradient, is_adversarial = adversarial.predictions_and_gradient(image, label)  # noqa: E501
    assert (predictions == first_predictions).all()
    assert gradient.shape == image.shape
    assert is_adversarial

    predictions, gradient, is_adversarial = adversarial.predictions_and_gradient()  # noqa: E501
    assert (predictions == first_predictions).all()
    assert gradient.shape == image.shape
    assert is_adversarial

    gradient = adversarial.gradient()
    assert gradient.shape == image.shape
    assert is_adversarial

    assert adversarial.num_classes() == 1000

    assert adversarial.has_gradient()

    assert adversarial.channel_axis(batch=True) == 3
    assert adversarial.channel_axis(batch=False) == 2

    # without adversarials
    criterion.is_adversarial = Mock(return_value=False)
    adversarial = Adversarial(model, criterion, image, label)
    predictions, is_adversarial, index = adversarial.batch_predictions(image[np.newaxis], increasing=True)  # noqa: E501
    assert (predictions == first_predictions[np.newaxis]).all()
    assert not is_adversarial
    assert index is None

    # without gradient
    del model.predictions_and_gradient

    assert not adversarial.has_gradient()
