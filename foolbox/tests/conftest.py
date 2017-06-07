from unittest.mock import Mock
from os.path import join
from os.path import dirname
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image
import tensorflow as tf

from foolbox.criteria import Misclassification
from foolbox.criteria import TargetClass
from foolbox.criteria import OriginalClassProbability
from foolbox.models import TensorFlowModel
from foolbox.models.wrappers import GradientLess
from foolbox import Adversarial


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


@pytest.fixture
def bn_model():
    """Creates a simple brightness model that does not require training.

    """

    bounds = (0, 1)
    channel_axis = 3
    channels = 10  # == num_classes

    def mean_brightness_net(images):
        logits = tf.reduce_mean(images, axis=(1, 2))
        return logits

    images = tf.placeholder(tf.float32, (None, 5, 5, channels))
    logits = mean_brightness_net(images)

    with tf.Session():
        model = TensorFlowModel(
            images,
            logits,
            bounds=bounds,
            channel_axis=channel_axis)

        yield model


@pytest.fixture
def gl_bn_model():
    """Same as bn_model but without gradient.

    """
    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        model = GradientLess(model)
        yield model


@pytest.fixture
def bn_image():
    np.random.seed(22)
    image = np.random.uniform(size=(5, 5, 10)).astype(np.float32)
    return image


@pytest.fixture
def bn_label():
    image = bn_image()
    mean = np.mean(image, axis=(0, 1))
    assert mean.shape == (10,)
    label = np.argmax(mean)
    return label


@pytest.fixture
def bn_criterion():
    return Misclassification()


@pytest.fixture
def bn_targeted_criterion():
    label = bn_label()
    assert label in [0, 1]
    return TargetClass(1 - label)


@pytest.fixture
def bn_impossible_criterion():
    """Does not consider any image as adversarial."""
    return OriginalClassProbability(0.)


@pytest.fixture
def bn_trivial_criterion():
    """Does consider every image as adversarial."""
    return OriginalClassProbability(1.)


@pytest.fixture
def bn_adversarial():
    criterion = bn_criterion()
    image = bn_image()
    label = bn_label()

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def bn_targeted_adversarial():
    criterion = bn_targeted_criterion()
    image = bn_image()
    label = bn_label()

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def gl_bn_adversarial():
    criterion = bn_criterion()
    image = bn_image()
    label = bn_label()

    cm_model = contextmanager(gl_bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def bn_impossible():
    criterion = bn_impossible_criterion()
    image = bn_image()
    label = bn_label()

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def bn_trivial():
    criterion = bn_trivial_criterion()
    image = bn_image()
    label = bn_label()

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)
