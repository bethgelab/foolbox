import pytest
import numpy as np

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
    with model:
        assert model.num_classes() == 0


def test_differentiable_base_model():

    class TestModel(models.DifferentiableModel):

        def batch_predictions(self, images):
            pass

        def num_classes(self):
            return 10

        def predictions_and_gradient(self, image, label):
            return 'predictions', 'gradient'

    model = TestModel(bounds=(0, 1), channel_axis=1)

    image = np.ones((28, 28, 1), dtype=np.float32)
    label = 2
    assert model.gradient(image, label) == 'gradient'
