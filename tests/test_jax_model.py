import pytest
import numpy as np
import jax.numpy as jnp
import eagerpy as ep

from foolbox.ext.native.models import JAXModel
from foolbox.ext.native.attacks import LinfinityBasicIterativeAttack
from foolbox.ext.native.attacks import L2BasicIterativeAttack
from foolbox.ext.native.attacks import L2CarliniWagnerAttack
from foolbox.ext.native.attacks import PGD


@pytest.fixture
def fmodel():
    bounds = (0, 1)

    def model(x):
        return jnp.mean(x, axis=(1, 2))

    fmodel = JAXModel(model, bounds=bounds)
    return fmodel


@pytest.fixture
def fmodel_and_data(fmodel):
    channels = num_classes = 10
    batch_size = 8
    h = w = 32

    np.random.seed(0)
    x = np.random.uniform(*fmodel.bounds(), size=(batch_size, h, w, channels)).astype(
        np.float32
    )
    x = jnp.asarray(x)

    y = np.arange(batch_size) % num_classes
    y = jnp.asarray(y)
    return fmodel, x, y, batch_size, num_classes


def test_jax_model_forward(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    output = fmodel.forward(x)
    assert output.shape == (batch_size, num_classes)
    assert isinstance(output, jnp.ndarray)


def test_jax_model_gradient(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    output = fmodel.gradient(x, y)
    assert output.shape == x.shape
    assert isinstance(output, jnp.ndarray)


def test_jax_linf_basic_iterative_attack(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = ep.astensor(fmodel.forward(x)).argmax(axis=-1)

    attack = LinfinityBasicIterativeAttack(fmodel)
    advs = attack(x, y, rescale=False, epsilon=0.3)

    perturbation = ep.astensor(advs - x)
    y_advs = ep.astensor(fmodel.forward(advs)).argmax(axis=-1)

    assert x.shape == advs.shape
    assert perturbation.abs().max() <= 0.3 + 1e-7
    assert (y_advs == y).float32().mean() < 1


def test_jax_l2_basic_iterative_attack(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = ep.astensor(fmodel.forward(x)).argmax(axis=-1)

    attack = L2BasicIterativeAttack(fmodel)
    advs = attack(x, y, rescale=False, epsilon=0.3)

    perturbation = ep.astensor(advs - x)
    y_advs = ep.astensor(fmodel.forward(advs)).argmax(axis=-1)

    assert x.shape == advs.shape
    assert perturbation.abs().max() <= 0.3 + 1e-7
    assert (y_advs == y).float32().mean() < 1


def test_jax_l2_carlini_wagner_attack(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = ep.astensor(fmodel.forward(x)).argmax(axis=-1)

    attack = L2CarliniWagnerAttack(fmodel)
    advs = attack(x, y, max_iterations=100)

    y_advs = ep.astensor(fmodel.forward(advs)).argmax(axis=-1)

    assert x.shape == advs.shape
    assert (y_advs == y).float32().mean() < 1


def test_jax_linf_pgd(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = ep.astensor(fmodel.forward(x)).argmax(axis=-1)

    attack = PGD(fmodel)
    advs = attack(x, y, rescale=False, epsilon=0.3)

    perturbation = ep.astensor(advs - x)
    y_advs = ep.astensor(fmodel.forward(advs)).argmax(axis=-1)

    assert x.shape == advs.shape
    assert perturbation.abs().max() <= 0.3 + 1e-7
    assert (y_advs == y).float32().mean() < 1
