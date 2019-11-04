# the different frameworks interfer with each other and
# sometimes cause segfaults or similar problems;
# choosing the right import order seems to be a
# workaround; given the current test order,
# first import tensorflow, then pytorch and then
# according to test order seems to solve it
import tensorflow

# import theano
# import mxnet
# import keras
import torch
import jax

from unittest.mock import Mock
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

from foolbox.models import CaffeModel
from foolbox.models import ModelWithoutGradients
from foolbox.models import ModelWithEstimatedGradients
from foolbox.models import EnsembleAveragedModel
from foolbox.v1 import Adversarial
from foolbox.distances import MSE
from foolbox.distances import Linfinity
from foolbox.distances import MAE
from foolbox.gradient_estimators import CoordinateWiseGradientEstimator
from foolbox.gradient_estimators import EvolutionaryStrategiesGradientEstimator
from foolbox.utils import binarize

import logging

logging.getLogger().setLevel(logging.DEBUG)

print(tensorflow.__version__)
# print(theano.__version__)
# print(mxnet.__version__)
# print(keras.__version__)
print(torch.__version__)
print(jax.__version__)


@pytest.fixture
def image():
    image = Image.open(join(dirname(__file__), "data/example.jpg"))
    image = np.asarray(image, dtype=np.float32)
    assert image.shape == (224, 224, 3)
    return image


@pytest.fixture
def label():
    return 333


@pytest.fixture
def model(image):
    predictions = np.array([1.0, 0.0, 0.5] * 111 + [2.0] + [0.3, 0.5, 1.1] * 222)
    model = Mock()
    model.bounds = Mock(return_value=(0, 255))
    model.forward_one = Mock(return_value=predictions)
    model.forward = Mock(return_value=predictions[np.newaxis])
    gradient = image
    model.forward_and_gradient_one = Mock(return_value=(predictions, gradient))
    model.forward_and_gradient = lambda inputs, _: (
        np.array([predictions] * len(inputs)),
        np.array([gradient] * len(inputs)),
    )
    model.gradient_one = Mock(return_value=gradient)
    model.backward_one = Mock(return_value=gradient)
    model.gradient = Mock(return_value=gradient[np.newaxis])
    model.backward = Mock(return_value=gradient[np.newaxis])
    model.num_classes = Mock(return_value=1000)
    model.channel_axis = Mock(return_value=3)
    return model


@pytest.fixture
def criterion():
    return Misclassification()


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
            images, logits, bounds=bounds, channel_axis=channel_axis
        )

        yield model


# bn_model is also needed as a function, so we create the fixture separately
@pytest.fixture(name="bn_model")
def bn_model_fixture():
    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield model


def sn_bn_model():
    """Creates a simple brightness model that does not require training with stochastic
    noise to simulate the behaviour of e.g. bayesian networks.

    """

    import tensorflow as tf

    bounds = (0, 1)
    channel_axis = 3
    channels = 10  # == num_classes

    def mean_brightness_net(images):
        logits = tf.reduce_mean(images, axis=(1, 2)) + tf.reduce_mean(
            images * tf.random.normal(shape=(1, 1, 1, 1), stddev=1e-4), axis=(1, 2)
        )
        return logits

    images = tf.placeholder(tf.float32, (None, 5, 5, channels))
    logits = mean_brightness_net(images)

    with tf.Session():
        model = TensorFlowModel(
            images, logits, bounds=bounds, channel_axis=channel_axis
        )

        yield model


# sn_bn_model is also needed as a function, so we create the fixture separately
@pytest.fixture(name="sn_bn_model")
def sn_bn_model_fixture():
    cm_model = contextmanager(sn_bn_model)
    with cm_model() as model:
        yield model


@pytest.fixture
def avg_sn_bn_adversarial(bn_criterion, bn_image, bn_label):
    criterion = bn_criterion
    image = bn_image
    label = bn_label

    cm_model = contextmanager(avg_bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


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
    model = PyTorchModel(model, bounds=bounds, num_classes=num_classes, device="cpu")
    return model


@pytest.fixture
def bn_model_caffe(request, tmpdir):
    """Same as bn_model but with Caffe."""

    import caffe
    from caffe import layers as L

    bounds = (0, 1)
    num_classes = channels = getattr(request, "param", 1000)

    net_spec = caffe.NetSpec()
    net_spec.data = L.Input(name="data", shape=dict(dim=[1, channels, 5, 5]))
    net_spec.reduce_1 = L.Reduction(
        net_spec.data, reduction_param={"operation": 4, "axis": 3}
    )
    net_spec.output = L.Reduction(
        net_spec.reduce_1, reduction_param={"operation": 4, "axis": 2}
    )
    net_spec.label = L.Input(name="label", shape=dict(dim=[1]))
    net_spec.loss = L.SoftmaxWithLoss(net_spec.output, net_spec.label)
    wf = tmpdir.mkdir("test_models_caffe_fixture").join(
        "test_caffe_{}.prototxt".format(num_classes)
    )
    wf.write("force_backward: true\n" + str(net_spec.to_proto()))
    net = caffe.Net(str(wf), caffe.TEST)
    model = CaffeModel(net, bounds=bounds)
    return model


def gl_bn_model():
    """Same as bn_model but without gradient.

    """
    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        model = ModelWithoutGradients(model)
        yield model


# gl_bn_model is also needed as a function, so we create the fixture separately
@pytest.fixture(name="gl_bn_model")
def gl_bn_model_fixture():
    cm_model = contextmanager(gl_bn_model)
    with cm_model() as model:
        yield model


def avg_bn_model():
    """Same as bn_model but without gradient.

    """
    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        model = EnsembleAveragedModel(model, ensemble_size=2)
        yield model


# avg_bn_model is also needed as a function, so we create the fixture separately
@pytest.fixture(name="avg_bn_model")
def avg_bn_model_fixture():
    cm_model = contextmanager(avg_bn_model)
    with cm_model() as model:
        yield model


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


# eg_bn_model_factory is also needed as a function, so we create the
# fixture separately
@pytest.fixture(
    name="eg_bn_model_factory",
    params=[CoordinateWiseGradientEstimator, EvolutionaryStrategiesGradientEstimator],
)
def eg_bn_model_factory_fixture(request):
    return eg_bn_model_factory(request)


@pytest.fixture
def bn_image():
    np.random.seed(22)
    image = np.random.uniform(size=(5, 5, 10)).astype(np.float32)
    return image


@pytest.fixture
def bn_images():
    np.random.seed(22)
    # To port the existing unit test to the batched mode we use the same
    # random image multiple times
    # TODO: Adjust the parameters in the unit test so that a batch of multiple
    #   random images can be used
    image = np.random.uniform(size=(1, 5, 5, 10)).astype(np.float32)
    images = np.repeat(image, 7, axis=0)
    return images


@pytest.fixture
def bn_image_pytorch():
    np.random.seed(22)
    image = np.random.uniform(size=(10, 5, 5)).astype(np.float32)
    return image


@pytest.fixture
def bn_label(bn_image):
    image = bn_image
    mean = np.mean(image, axis=(0, 1))
    assert mean.shape == (10,)
    label = np.argmax(mean)
    return label


@pytest.fixture
def bn_labels(bn_images):
    images = bn_images
    mean = np.mean(images, axis=(1, 2))
    labels = np.argmax(mean, axis=-1)
    return labels


@pytest.fixture
def bn_label_pytorch(bn_image_pytorch):
    image = bn_image_pytorch
    mean = np.mean(image, axis=(1, 2))
    assert mean.shape == (10,)
    label = np.argmax(mean)
    return label


@pytest.fixture
def bn_criterion():
    return Misclassification()


@pytest.fixture
def bn_targeted_criterion(bn_label):
    label = bn_label
    assert label in [0, 1]
    return TargetClass(1 - label)


@pytest.fixture
def bn_impossible_criterion():
    """Does not consider any image as adversarial."""
    return OriginalClassProbability(0.0)


@pytest.fixture
def bn_trivial_criterion():
    """Does consider every image as adversarial."""
    return OriginalClassProbability(1.0)


@pytest.fixture
def bn_adversarial(bn_criterion, bn_image, bn_label):
    criterion = bn_criterion
    image = bn_image
    label = bn_label

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def bn_adversarial_linf(bn_criterion, bn_image, bn_label):
    criterion = bn_criterion
    image = bn_image
    label = bn_label
    distance = Linfinity

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label, distance=distance)


@pytest.fixture
def bn_adversarial_mae(bn_criterion, bn_image, bn_label):
    criterion = bn_criterion
    image = bn_image
    label = bn_label
    distance = MAE

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label, distance=distance)


@pytest.fixture
def bn_targeted_adversarial(bn_targeted_criterion, bn_image, bn_label):
    criterion = bn_targeted_criterion
    image = bn_image
    label = bn_label

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def gl_bn_adversarial(bn_criterion, bn_image, bn_label):
    criterion = bn_criterion
    image = bn_image
    label = bn_label

    cm_model = contextmanager(gl_bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture(
    params=[CoordinateWiseGradientEstimator, EvolutionaryStrategiesGradientEstimator]
)
def eg_bn_adversarial(request, bn_criterion, bn_image, bn_label):
    criterion = bn_criterion
    image = bn_image
    label = bn_label

    eg_bn_model = eg_bn_model_factory(request)

    cm_model = contextmanager(eg_bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture(
    params=[CoordinateWiseGradientEstimator, EvolutionaryStrategiesGradientEstimator]
)
def eg_bn_model(request):
    eg_bn_model = eg_bn_model_factory(request)

    cm_model = contextmanager(eg_bn_model)
    with cm_model() as model:
        yield model


@pytest.fixture
def bn_impossible(bn_impossible_criterion, bn_image, bn_label):
    criterion = bn_impossible_criterion
    image = bn_image
    label = bn_label

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def bn_trivial(bn_trivial_criterion, bn_image, bn_label):
    criterion = bn_trivial_criterion
    image = bn_image
    label = bn_label

    cm_model = contextmanager(bn_model)
    with cm_model() as model:
        adv = Adversarial(model, criterion, image, label)
        # the original should not yet be considered adversarial
        # so that the attack implementation is actually called
        adv._Adversarial__best_adversarial = None
        adv._Adversarial__best_distance = MSE(value=np.inf)
        yield adv


@pytest.fixture
def bn_adversarial_pytorch(
    bn_model_pytorch, bn_criterion, bn_image_pytorch, bn_label_pytorch
):
    model = bn_model_pytorch
    criterion = bn_criterion
    image = bn_image_pytorch
    label = bn_label_pytorch
    adv = Adversarial(model, criterion, image, label)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf
    return adv


@pytest.fixture
def bn_targeted_adversarial_pytorch(
    bn_model_pytorch, bn_targeted_criterion, bn_image_pytorch, bn_label_pytorch
):
    model = bn_model_pytorch
    criterion = bn_targeted_criterion
    image = bn_image_pytorch
    label = bn_label_pytorch
    adv = Adversarial(model, criterion, image, label)
    assert adv.perturbed is None
    assert adv.distance.value == np.inf
    return adv


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
            preprocessing=preprocessing,
        )

        yield model


# binarized_bn_model is also needed as a function, so we create the
# fixture separately
@pytest.fixture(name="binarized_bn_model")
def binarized_bn_model_fixture():
    cm_model = contextmanager(binarized_bn_model)
    with cm_model() as model:
        yield model


@pytest.fixture
def binarized_bn_adversarial(bn_criterion, bn_image, binarized_bn_label):
    criterion = bn_criterion
    image = bn_image
    label = binarized_bn_label

    cm_model = contextmanager(binarized_bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def binarized_bn_label(bn_image):
    image = bn_image
    image = binarize(image, (0, 1))
    mean = np.mean(image, axis=(0, 1))
    assert mean.shape == (10,)
    label = np.argmax(mean)
    return label


@pytest.fixture
def binarized_bn_labels(bn_images):
    images = bn_images
    images = binarize(images, (0, 1))
    means = np.mean(images, axis=(1, 2))
    assert means.shape == (len(bn_images), 10)
    labels = np.argmax(means, axis=-1)
    assert labels.shape == (len(bn_images),)
    return labels


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
        x = binarize(x, (0, 1), included_in="lower")

        def backward(x):
            return x

        return x, backward

    with tf.Session():
        model = TensorFlowModel(
            images,
            logits,
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing,
        )

        yield model


# binarized2_bn_model is also needed as a function, so we create the
# fixture separately
@pytest.fixture(name="binarized2_bn_model")
def binarized2_bn_model_fixture():
    cm_model = contextmanager(binarized2_bn_model)
    with cm_model() as model:
        yield model


@pytest.fixture
def binarized2_bn_adversarial(bn_criterion, bn_image, binarized2_bn_label):
    criterion = bn_criterion
    image = bn_image
    label = binarized2_bn_label

    cm_model = contextmanager(binarized2_bn_model)
    with cm_model() as model:
        yield Adversarial(model, criterion, image, label)


@pytest.fixture
def binarized2_bn_label(bn_image):
    image = bn_image
    image = binarize(image, (0, 1), included_in="lower")
    mean = np.mean(image, axis=(0, 1))
    assert mean.shape == (10,)
    label = np.argmax(mean)
    return label


@pytest.fixture
def binarized2_bn_labels(bn_images):
    images = bn_images
    images = binarize(images, (0, 1), included_in="lower")
    means = np.mean(images, axis=(1, 2))
    assert means.shape == (len(images), 10)
    labels = np.argmax(means, -1)
    return labels
