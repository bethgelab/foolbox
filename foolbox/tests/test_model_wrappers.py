import numpy as np

from foolbox.models import ModelWrapper
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
