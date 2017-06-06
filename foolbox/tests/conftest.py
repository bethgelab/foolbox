from unittest.mock import Mock
from os.path import join
from os.path import dirname

import numpy as np
import pytest
from PIL import Image

from foolbox.criteria import Misclassification


@pytest.fixture
def image():
    image = Image.open(join(dirname(__file__), 'data/example.jpg'))
    image = np.asarray(image, dtype=np.float32)
    assert image.shape == (224, 224, 3)
    return image


@pytest.fixture
def label():
    return 914


@pytest.fixture
def model():
    predictions = np.array([1., 0., 0.5] * 111 + [2.] + [0.3, 0.5, 1.1] * 222)
    model = Mock()
    model.bounds = Mock(return_value=(0, 255))
    model.predictions = Mock(return_value=predictions)
    model.batch_predictions = Mock(return_value=predictions[np.newaxis])
    gradient = image()
    model.predictions_and_gradient = Mock(return_value=(predictions, gradient))  # noqa: E501
    model.gradient = Mock(return_value=gradient)  # noqa: E501
    model.num_classes = Mock(return_value=1000)
    model.channel_axis = Mock(return_value=3)
    return model


@pytest.fixture
def criterion():
    return Misclassification()
