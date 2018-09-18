# the different frameworks interfer with each other and
# sometimes cause segfaults or similar problems;
# choosing the right import order seems to be a
# workaround; given the current test order,
# first import tensorflow, then pytorch and then
# according to test order seems to solve it
import tensorflow
print(tensorflow.__version__)
# import theano
# print(theano.__version__)
# import mxnet
# print(mxnet.__version__)
# import keras
# print(keras.__version__)
import torch
print(torch.__version__)


import sys
if sys.version_info > (3, 2):
    from unittest.mock import Mock
else:
    # for Python2.7 compatibility
    from mock import Mock

from os.path import join
from os.path import dirname
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from foolbox.criteria import Misclassification
from foolbox.criteria import TargetClass
from foolbox.criteria import OriginalClassProbability
from foolbox.models import TensorFlowModel
from foolbox.models import PyTorchModel
from foolbox.models import ModelWithoutGradients
from foolbox.models import ModelWithEstimatedGradients
from foolbox import Adversarial
from foolbox.distances import MSE
from foolbox.distances import Linfinity
from foolbox.distances import MAE
from foolbox.gradient_estimators import CoordinateWiseGradientEstimator
from foolbox.gradient_estimators import EvolutionaryStrategiesGradientEstimator
from foolbox.utils import binarize


@pytest.fixture
def image():
    image = Image.open(join(dirname(__file__), 'data/example.jpg'))
    image = np.asarray(image, dtype=np.float32)
    assert image.shape == (224, 224, 3)
    return image


@pytest.fixture
def label():
    return 333


@pytest.fixture
def model():
    predictions = np.array([1., 0., 0.5] * 111 + [2.] + [0.3, 0.5, 1.1] * 222)
    model = Mock()
    model.bounds = Mock(return_value=(0, 255))
    model.predictions = Mock(return_value=predictions)
    model.batch_predictions = Mock(return_value=predictions[np.newaxis])
    gradient = image()
    model.predictions_and_gradient = Mock(return_value=(predictions, gradient))  # noqa: E501
    model.gradient = Mock(return_value=gradient)
    model.backward = Mock(return_value=gradient)
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

    import tensorflow as tf

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
def bn_model_pytorch():
    """Same as bn_model but with PyTorch."""

    import torch
    import torch.nn as nn

    bounds = (0, 1)
    num_classes = 10

    class Net(nn.Module):

        def forward(self, x):
            assert isinstance(x.data, torch.FloatTensor)
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            logits = x
            return logits

    model = Net()
    model = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        device='cpu')
    return model


@pytest.fixture
def gl_bn_model():
    """Same as bn_model but without gradient.

    """
    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        model = ModelWithoutGradients(model)
        yield model


@pytest.fixture(params=[CoordinateWiseGradientEstimator,
                        EvolutionaryStrategiesGradientEstimator])
def eg_bn_model_factory(request):
    """Same as bn_model but with estimated gradient.

    """
    GradientEstimator = request.param

    def eg_bn_model():
        cm_model = contextmanager(bn_model)
        with cm_model() as model:
            gradient_estimator = GradientEstimator(epsilon=0.01)
            model = ModelWithEstimatedGradients(model, gradient_estimator)
            yield model
    return eg_bn_model


@pytest.fixture
def bn_image():
    np.random.seed(22)
    image = np.random.uniform(size=(5, 5, 10)).astype(np.float32)
    return image


@pytest.fixture
def bn_image_pytorch():
    np.random.seed(22)
    image = np.random.uniform(size=(10, 5, 5)).astype(np.float32)
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
def bn_adversarial_linf():
    criterion = bn_criterion()
    image = bn_image()
    label = bn_label()
    distance = Linfinity

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label, distance=distance)


@pytest.fixture
def bn_adversarial_mae():
    criterion = bn_criterion()
    image = bn_image()
    label = bn_label()
    distance = MAE

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label, distance=distance)


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


@pytest.fixture(params=[CoordinateWiseGradientEstimator,
                        EvolutionaryStrategiesGradientEstimator])
def eg_bn_adversarial(request):
    criterion = bn_criterion()
    image = bn_image()
    label = bn_label()

    eg_bn_model = eg_bn_model_factory(request)

    cm_model = contextmanager(eg_bn_model)
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
        adv = Adversarial(model, criterion, image, label)
        # the original should not yet be considered adversarial
        # so that the attack implementation is actually called
        adv._Adversarial__best_adversarial = None
        adv._Adversarial__best_distance = MSE(value=np.inf)
        yield adv


@pytest.fixture
def bn_adversarial_pytorch():
    model = bn_model_pytorch()
    criterion = bn_criterion()
    image = bn_image_pytorch()
    label = bn_label()
    return Adversarial(model, criterion, image, label)


@pytest.fixture
def bn_targeted_adversarial_pytorch():
    model = bn_model_pytorch()
    criterion = bn_targeted_criterion()
    image = bn_image_pytorch()
    label = bn_label()
    return Adversarial(model, criterion, image, label)


@pytest.fixture
def binarized_bn_model():
    """Creates a simple brightness model that does not require training.

    """

    import tensorflow as tf

    bounds = (0, 1)
    channel_axis = 3
    channels = 10  # == num_classes

    def mean_brightness_net(images):
        logits = tf.reduce_mean(images, axis=(1, 2))
        return logits

    images = tf.placeholder(tf.float32, (None, 5, 5, channels))
    logits = mean_brightness_net(images)

    def preprocessing(x):
        x = binarize(x, (0, 1))

        def backward(x):
            return x
        return x, backward

    with tf.Session():
        model = TensorFlowModel(
            images,
            logits,
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        yield model


@pytest.fixture
def binarized_bn_adversarial():
    criterion = bn_criterion()
    image = bn_image()
    label = binarized_bn_label()

    cm_model = contextmanager(binarized_bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def binarized_bn_label():
    image = bn_image()
    image = binarize(image, (0, 1))
    mean = np.mean(image, axis=(0, 1))
    assert mean.shape == (10,)
    label = np.argmax(mean)
    return label


@pytest.fixture
def binarized2_bn_model():
    """Creates a simple brightness model that does not require training.

    """

    import tensorflow as tf

    bounds = (0, 1)
    channel_axis = 3
    channels = 10  # == num_classes

    def mean_brightness_net(images):
        logits = tf.reduce_mean(images, axis=(1, 2))
        return logits

    images = tf.placeholder(tf.float32, (None, 5, 5, channels))
    logits = mean_brightness_net(images)

    def preprocessing(x):
        x = binarize(x, (0, 1), included_in='lower')

        def backward(x):
            return x
        return x, backward

    with tf.Session():
        model = TensorFlowModel(
            images,
            logits,
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        yield model


@pytest.fixture
def binarized2_bn_adversarial():
    criterion = bn_criterion()
    image = bn_image()
    label = binarized2_bn_label()

    cm_model = contextmanager(binarized2_bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def binarized2_bn_label():
    image = bn_image()
    image = binarize(image, (0, 1), included_in='lower')
    mean = np.mean(image, axis=(0, 1))
    assert mean.shape == (10,)
    label = np.argmax(mean)
    return label
