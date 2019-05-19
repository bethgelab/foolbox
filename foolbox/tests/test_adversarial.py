import numpy as np

from foolbox import Adversarial
from foolbox.distances import MSE
import foolbox

import sys
if sys.version_info > (3, 2):
    from unittest.mock import Mock
else:
    # for Python2.7 compatibility
    from mock import Mock


# def test_adversarial(bn_model, bn_criterion, bn_image, bn_label):
def test_adversarial(model, criterion, image, label):
    # model = bn_model
    # criterion = bn_criterion
    # image = bn_image
    # label = bn_label

    adversarial = Adversarial(model, criterion, image, label, verbose=False)

    assert not adversarial.forward_one(image)[1]

    assert adversarial.perturbed is None
    assert adversarial.output is None
    assert adversarial.adversarial_class is None
    assert adversarial.distance == MSE(value=np.inf)
    assert adversarial.unperturbed is image
    assert adversarial.original_class == label
    assert adversarial.target_class() is None
    assert adversarial.normalized_distance(image) == MSE(value=0)
    assert adversarial.normalized_distance(image).value == 0

    np.random.seed(22)
    perturbation = np.random.uniform(-1, 1, size=image.shape)
    perturbed = np.clip(image + perturbation, 0, 255).astype(np.float32)
    d1 = adversarial.normalized_distance(perturbed).value
    assert d1 != 0

    assert adversarial.unperturbed.dtype == np.float32

    adversarial.set_distance_dtype(np.float32)
    assert adversarial.normalized_distance(perturbed).value == d1

    adversarial.set_distance_dtype(np.float64)
    assert adversarial.normalized_distance(perturbed).value != d1

    adversarial.reset_distance_dtype()
    assert adversarial.normalized_distance(perturbed).value == d1

    true_label = label
    label = 22  # wrong label
    adversarial = Adversarial(model, criterion, image, label, verbose=True)

    assert adversarial.perturbed is not None
    assert adversarial.output is not None
    assert adversarial.adversarial_class == true_label
    assert adversarial.adversarial_class == np.argmax(adversarial.output)
    assert adversarial.distance == MSE(value=0)
    assert adversarial.unperturbed is image
    assert adversarial.original_class == label
    assert adversarial.target_class() is None
    assert adversarial.normalized_distance(image) == MSE(value=0)
    assert adversarial.normalized_distance(image).value == 0

    predictions, is_adversarial = adversarial.forward_one(image)
    first_predictions = predictions
    assert is_adversarial

    predictions, is_adversarial, _, _ = adversarial.forward_one(image, return_details=True)
    first_predictions = predictions
    assert is_adversarial

    predictions, is_adversarial = adversarial.forward(image[np.newaxis])
    assert (predictions == first_predictions[np.newaxis]).all()
    assert np.all(is_adversarial == np.array([True]))

    predictions, is_adversarial, index = adversarial.forward(image[np.newaxis], greedy=True)
    assert (predictions == first_predictions[np.newaxis]).all()
    assert is_adversarial
    assert index == 0

    predictions, is_adversarial, index, _, _ = adversarial.forward(image[np.newaxis], greedy=True, return_details=True)
    assert (predictions == first_predictions[np.newaxis]).all()
    assert is_adversarial
    assert index == 0

    predictions, gradient, is_adversarial = adversarial.forward_and_gradient_one(image, label)
    assert (predictions == first_predictions).all()
    assert gradient.shape == image.shape
    assert is_adversarial

    predictions, gradient, is_adversarial, _, _ = adversarial.forward_and_gradient_one(
        image, label, return_details=True)
    assert (predictions == first_predictions).all()
    assert gradient.shape == image.shape
    assert is_adversarial

    predictions, gradient, is_adversarial = adversarial.forward_and_gradient_one()
    assert (predictions == first_predictions).all()
    assert gradient.shape == image.shape
    assert is_adversarial

    gradient_pre = np.ones_like(predictions) * 0.3
    gradient = adversarial.backward_one(gradient_pre, image)
    gradient2 = adversarial.backward_one(gradient_pre)
    assert gradient.shape == image.shape
    assert (gradient == gradient2).all()

    gradient = adversarial.gradient_one()
    assert gradient.shape == image.shape
    assert is_adversarial

    assert adversarial.num_classes() == 1000

    assert adversarial.has_gradient()

    assert adversarial.channel_axis(batch=True) == 3
    assert adversarial.channel_axis(batch=False) == 2

    # without adversarials
    criterion.is_adversarial = Mock(return_value=False)
    adversarial = Adversarial(model, criterion, image, label)
    predictions, is_adversarial, index = adversarial.forward(image[np.newaxis], greedy=True)
    assert (predictions == first_predictions[np.newaxis]).all()
    assert not is_adversarial
    assert index is None

    del model.gradient
    assert not adversarial.has_gradient()


def test_inplace(bn_model, bn_adversarial, bn_label):
    class TestAttack(foolbox.attacks.Attack):
        @foolbox.attacks.base.call_decorator
        def __call__(self, input_or_adv, label, unpack):
            a = input_or_adv
            x = np.zeros_like(a.unperturbed)
            a.forward_one(x)
            x[:] = a.unperturbed

    assert bn_adversarial.perturbed is None
    assert np.argmax(bn_model.forward_one(bn_adversarial.unperturbed)) == bn_label
    attack = TestAttack()
    attack(bn_adversarial)
    assert bn_adversarial.perturbed is not None
    assert bn_adversarial.distance.value > 0
    assert np.argmax(bn_model.forward_one(bn_adversarial.unperturbed)) == bn_label
    assert np.argmax(bn_model.forward_one(bn_adversarial.perturbed)) != bn_label
    assert not (bn_adversarial.perturbed == bn_adversarial.unperturbed).all()
    assert (bn_adversarial.distance.reference == bn_adversarial.unperturbed).all()
    assert (bn_adversarial.distance.other == bn_adversarial.perturbed).all()
