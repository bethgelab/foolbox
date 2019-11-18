import pytest
import numpy as np
import tensorflow as tf
import eagerpy as ep
from itertools import product

from foolbox.ext.native.models import TensorFlowModel
from foolbox.ext.native.attacks import LinfinityBasicIterativeAttack
from foolbox.ext.native.attacks import L2BasicIterativeAttack
from foolbox.ext.native.attacks import L2CarliniWagnerAttack
from foolbox.ext.native.attacks import PGD


def fmodel_sequential():
    bounds = (0, 1)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    fmodel = TensorFlowModel(model, bounds=bounds)
    return fmodel


def fmodel_subclassing():
    bounds = (0, 1)

    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.pool = tf.keras.layers.GlobalAveragePooling2D()

        def call(self, x):
            x = self.pool(x)
            return x

    model = Model()
    fmodel = TensorFlowModel(model, bounds=bounds)
    return fmodel


def fmodel_functional():
    bounds = (0, 1)
    channels = 10
    h = w = 32

    if tf.keras.backend.image_data_format() == "channels_first":
        x = x_ = tf.keras.Input(shape=(channels, h, w))
    else:
        x = x_ = tf.keras.Input(shape=(h, w, channels))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=x_, outputs=x)
    fmodel = TensorFlowModel(model, bounds=bounds)
    return fmodel


fmodel_and_data_params = list(
    product(
        ["channels_first", "channels_last"],
        ["sequential", "subclassing", "functional"],
    )
)


@pytest.fixture(
    params=fmodel_and_data_params,
    ids=list(map(lambda x: ", ".join(x), fmodel_and_data_params)),
)
def fmodel_and_data(request):
    data_format, model_api = request.param
    tf.keras.backend.set_image_data_format(data_format)
    fmodel = {
        "sequential": fmodel_sequential,
        "subclassing": fmodel_subclassing,
        "functional": fmodel_functional,
    }[model_api]()

    channels = num_classes = 10
    batch_size = 8
    h = w = 32

    if data_format == "channels_first":
        shape = (batch_size, channels, h, w)
    else:
        shape = (batch_size, h, w, channels)

    np.random.seed(0)
    x = np.random.uniform(*fmodel.bounds(), shape).astype(np.float32)
    x = tf.constant(x)

    y = np.arange(batch_size) % num_classes
    y = tf.constant(y)
    return fmodel, x, y, batch_size, num_classes


def test_tensorflow_model_forward(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    output = fmodel.forward(x)
    assert output.shape == (batch_size, num_classes)
    assert isinstance(output, tf.Tensor)


def test_tensorflow_model_gradient(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    output = fmodel.gradient(x, y)
    assert output.shape == x.shape
    assert isinstance(output, tf.Tensor)


def test_tensorflow_linf_basic_iterative_attack(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = ep.astensor(fmodel.forward(x)).argmax(axis=-1)

    attack = LinfinityBasicIterativeAttack(fmodel)
    advs = attack(x, y, rescale=False, epsilon=0.3)

    perturbation = ep.astensor(advs - x)
    y_advs = ep.astensor(fmodel.forward(advs)).argmax(axis=-1)

    assert x.shape == advs.shape
    assert perturbation.abs().max() <= 0.3 + 1e-7
    assert (y_advs == y).float32().mean() < 1


def test_tensorflow_l2_basic_iterative_attack(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = ep.astensor(fmodel.forward(x)).argmax(axis=-1)

    attack = L2BasicIterativeAttack(fmodel)
    advs = attack(x, y, rescale=False, epsilon=0.3)

    perturbation = ep.astensor(advs - x)
    y_advs = ep.astensor(fmodel.forward(advs)).argmax(axis=-1)

    assert x.shape == advs.shape
    assert perturbation.abs().max() <= 0.3 + 1e-7
    assert (y_advs == y).float32().mean() < 1


def test_tensorflow_l2_carlini_wagner_attack(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = ep.astensor(fmodel.forward(x)).argmax(axis=-1)

    attack = L2CarliniWagnerAttack(fmodel)
    advs = attack(x, y, max_iterations=100)

    y_advs = ep.astensor(fmodel.forward(advs)).argmax(axis=-1)

    assert x.shape == advs.shape
    assert (y_advs == y).float32().mean() < 1


def test_tensorflow_linf_pgd(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = ep.astensor(fmodel.forward(x)).argmax(axis=-1)

    attack = PGD(fmodel)
    advs = attack(x, y, rescale=False, epsilon=0.3)

    perturbation = ep.astensor(advs - x)
    y_advs = ep.astensor(fmodel.forward(advs)).argmax(axis=-1)

    assert x.shape == advs.shape
    assert perturbation.abs().max() <= 0.3 + 1e-7
    assert (y_advs == y).float32().mean() < 1
