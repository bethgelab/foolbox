import sys
if sys.version_info > (3, 2):
    from unittest.mock import Mock
else:
    # for Python2.7 compatibility
    from mock import Mock

import numpy as np

from foolbox import Adversarial
from foolbox.distances import MSE


# def test_adversarial(bn_model, bn_criterion, bn_image, bn_label):
def test_adversarial(model, criterion, image, label):
    # model = bn_model
    # criterion = bn_criterion
    # image = bn_image
    # label = bn_label

    adversarial = Adversarial(model, criterion, image, label, verbose=False)

    assert not adversarial.predictions(image)[1]

    assert adversarial.image is None
    assert adversarial.distance == MSE(value=np.inf)
    assert adversarial.original_image is image
    assert adversarial.original_class == label
    assert adversarial.target_class() is None
    assert adversarial.normalized_distance(image) == MSE(value=0)
    assert adversarial.normalized_distance(image).value == 0

    label = 22  # wrong label
    adversarial = Adversarial(model, criterion, image, label, verbose=True)

    assert adversarial.image is not None
    assert adversarial.distance == MSE(value=0)
    assert adversarial.original_image is image
    assert adversarial.original_class == label
    assert adversarial.target_class() is None
    assert adversarial.normalized_distance(image) == MSE(value=0)
    assert adversarial.normalized_distance(image).value == 0

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

    gradient_pre = np.ones_like(predictions) * 0.3
    gradient = adversarial.backward(gradient_pre, image)
    gradient2 = adversarial.backward(gradient_pre)
    assert gradient.shape == image.shape
    assert (gradient == gradient2).all()

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
