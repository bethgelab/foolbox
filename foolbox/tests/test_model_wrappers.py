import numpy as np

from foolbox.models import ModelWrapper
from foolbox.models import DifferentiableModelWrapper
from foolbox.models import CompositeModel


def test_context_manager(gl_bn_model):
    assert isinstance(gl_bn_model, ModelWrapper)
    with gl_bn_model as model:
        assert model is not None
        assert isinstance(model, ModelWrapper)


def test_wrapping(gl_bn_model, bn_image):
    assert isinstance(gl_bn_model, ModelWrapper)
    assert gl_bn_model.num_classes() == 10
    assert np.all(
        gl_bn_model.predictions(bn_image) ==
        gl_bn_model.batch_predictions(bn_image[np.newaxis])[0])


def test_diff_wrapper(bn_model, bn_image, bn_label):
    x = bn_image
    la = bn_label
    xs = x[np.newaxis]
    model1 = bn_model
    model2 = DifferentiableModelWrapper(model1)
    assert model1.num_classes() == model2.num_classes()
    assert np.all(model1.predictions(x) == model2.predictions(x))
    assert np.all(model1.batch_predictions(xs) == model2.batch_predictions(xs))
    assert np.all(model1.gradient(x, la) == model2.gradient(x, la))
    assert np.all(model1.predictions_and_gradient(x, la)[0] ==
                  model2.predictions_and_gradient(x, la)[0])
    assert np.all(model1.predictions_and_gradient(x, la)[1] ==
                  model2.predictions_and_gradient(x, la)[1])
    g = model1.predictions(x)
    assert np.all(model1.backward(g, x) == model2.backward(g, x))


def test_composite_model(gl_bn_model, bn_model, bn_image, bn_label):
    model = CompositeModel(gl_bn_model, bn_model)
    with model:
        assert gl_bn_model.num_classes() == model.num_classes()
        assert np.all(
            gl_bn_model.predictions(bn_image) ==
            model.predictions(bn_image))
        assert np.all(
            bn_model.gradient(bn_image, bn_label) ==
            model.gradient(bn_image, bn_label))
        assert np.all(
            gl_bn_model.predictions(bn_image) ==
            model.predictions_and_gradient(bn_image, bn_label)[0])
        assert np.all(
            bn_model.predictions_and_gradient(bn_image, bn_label)[1] ==
            model.predictions_and_gradient(bn_image, bn_label)[1])


def test_estimate_gradient_wrapper(eg_bn_adversarial, bn_image):
    p, ia = eg_bn_adversarial.predictions(bn_image)
    np.random.seed(22)
    g = eg_bn_adversarial.gradient(bn_image)
    np.random.seed(22)
    p2, g2, ia2 = eg_bn_adversarial.predictions_and_gradient(bn_image)
    assert np.all(p == p2)
    assert np.all(g == g2)
    assert ia == ia2
