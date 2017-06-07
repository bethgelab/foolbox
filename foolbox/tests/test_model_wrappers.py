import numpy as np

from foolbox.models import ModelWrapper


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
