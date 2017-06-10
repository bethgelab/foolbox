from pytest import approx
import numpy as np

from foolbox.utils import softmax
from foolbox.utils import crossentropy
from foolbox.utils import imagenet_example


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
