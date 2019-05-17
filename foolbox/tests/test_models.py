import pytest

from foolbox import models


def test_abstract_model():
    with pytest.raises(TypeError):
        models.Model()


def test_abstract_differentiable_model():
    with pytest.raises(TypeError):
        models.DifferentiableModel()


def test_base_model():

    class TestModel(models.Model):

        def batch_predictions(self, images):
            pass

        def num_classes(self):
            return 0

    model = TestModel(bounds=(0, 1), channel_axis=1)
    assert model.bounds() == (0, 1)
    assert model.channel_axis() == 1
