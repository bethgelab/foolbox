import pytest
from pytest import approx
import numpy as np

from foolbox.utils import softmax
from foolbox.utils import crossentropy
from foolbox.utils import imagenet_example
from foolbox.utils import binarize
from foolbox.utils import onehot_like


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
