import pytest
from pytest import approx
import numpy as np

from foolbox.utils import softmax
from foolbox.utils import crossentropy
from foolbox.utils import imagenet_example
from foolbox.utils import binarize
from foolbox.utils import onehot_like
from foolbox.utils import samples


def test_softmax():
    predictions = np.array([0.1, 0.5, 0.7, 0.4])
    probabilities = softmax(predictions)
    assert not np.sum(predictions) == approx(1.)
    assert np.sum(probabilities) == approx(1.)


def test_crossentropy():
    predictions = np.array([0.1, 0.5, 0.7, 0.4])
    probabilities = softmax(predictions)
    for i in range(len(predictions)):
        ce = crossentropy(logits=predictions, label=i)
        assert ce == approx(-np.log(probabilities[i]))


def test_imagenet_example():
    image, label = imagenet_example()
    assert 0 <= label < 1000
    assert isinstance(label, int)
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.float32


def test_imagenet_example_channels_first():
    image, label = imagenet_example(data_format='channels_first')
    image2, _ = imagenet_example(data_format='channels_last')
    assert 0 <= label < 1000
    assert isinstance(label, int)
    assert image.shape == (3, 224, 224)
    assert image.dtype == np.float32

    for i in range(3):
        assert np.all(image[i] == image2[:, :, i])


def test_samples_imagenet():
    images, labels = samples(dataset='imagenet',
                             batchsize=5)
    assert 0 <= labels[0] < 1000
    assert images.shape[0] == 5
    assert isinstance(labels[0], np.integer)
    assert images.shape == (5, 224, 224, 3)
    assert images.dtype == np.float32


def test_samples_imagenet_channels_first():
    images, labels = samples(dataset='imagenet',
                             batchsize=5,
                             data_format='channels_first')
    assert 0 <= labels[0] < 1000
    assert images.shape[0] == 5
    assert isinstance(labels[0], np.integer)
    assert images.shape == (5, 3, 224, 224)
    assert images.dtype == np.float32


def test_samples_mnist():
    images, labels = samples(dataset='mnist', batchsize=5)
    assert 0 <= labels[0] < 10
    assert images.shape[0] == 5
    assert isinstance(labels[0], np.integer)
    assert images.shape == (5, 28, 28)
    assert images.dtype == np.float32


def test_samples_cifar10():
    images, labels = samples(dataset='cifar10', batchsize=5)
    assert 0 <= labels[0] < 10
    assert images.shape[0] == 5
    assert isinstance(labels[0], np.integer)
    assert images.shape == (5, 32, 32, 3)
    assert images.dtype == np.float32


def test_samples_cifar100():
    images, labels = samples(dataset='cifar100', batchsize=5)
    assert 0 <= labels[0] < 100
    assert images.shape[0] == 5
    assert isinstance(labels[0], np.integer)
    assert images.shape == (5, 32, 32, 3)
    assert images.dtype == np.float32


def test_samples_fashionMNIST():
    images, labels = samples(dataset='fashionMNIST', batchsize=5)
    assert 0 <= labels[0] < 10
    assert images.shape[0] == 5
    assert isinstance(labels[0], np.integer)
    assert images.shape == (5, 28, 28)
    assert images.dtype == np.float32


def test_binarize():
    x = np.array([0.1, 0.5, 0.7, 0.4])
    x1 = binarize(x, (-2, 2), 0.5)
    assert np.all(abs(x1) == 2)
    with pytest.raises(ValueError):
        binarize(x, (-2, 2), 0.5, included_in='blabla')


def test_onehot_like():
    a = np.array([0.1, 0.5, 0.7, 0.4])
    o = onehot_like(a, 2)
    assert o.shape == a.shape
    assert o.dtype == a.dtype
    assert np.all(o[:2] == 0)
    assert o[2] == 1
    assert np.all(o[3:] == 0)

    o = onehot_like(a, 3, value=-77.5)
    assert o.shape == a.shape
    assert o.dtype == a.dtype
    assert np.all(o[:3] == 0)
    assert o[3] == -77.5
    assert np.all(o[4:] == 0)
