import pytest
import numpy as np
from foolbox import distances
from pytest import approx


def test_abstract_distance():
    with pytest.raises(TypeError):
        distances.Distance()


def test_base_distance():

    class TestDistance(distances.Distance):

        def _calculate(self):
            return 22, 2

    distance = TestDistance(None, None, bounds=(0, 1))
    assert distance.name() == 'TestDistance'
    assert distance.value == 22
    assert distance.gradient == 2
    assert '2.2' in str(distance)
    assert 'TestDistance' in str(distance)
    assert distance == distance
    assert not distance < distance
    assert not distance > distance
    assert distance <= distance
    assert distance >= distance

    with pytest.raises(TypeError):
        distance < 3

    with pytest.raises(TypeError):
        distance == 3


def test_mse():
    assert distances.MSE == distances.MeanSquaredDistance


def test_mae():
    assert distances.MAE == distances.MeanAbsoluteDistance


def test_linf():
    assert distances.Linf == distances.Linfinity


def test_mean_squared_distance():
    d = distances.MeanSquaredDistance(
        np.array([0, .5]),
        np.array([.5, .5]),
        bounds=(0, 1))
    assert d.value == 1. / 8.
    assert (d.gradient == np.array([.5, 0])).all()


def test_mean_absolute_distance():
    d = distances.MeanAbsoluteDistance(
        np.array([0, .5]),
        np.array([.7, .5]),
        bounds=(0, 1))
    assert d.value == approx(0.35)
    assert (d.gradient == np.array([0.5, 0])).all()


def test_linfinity():
    d = distances.Linfinity(
        np.array([0, .5]),
        np.array([.7, .5]),
        bounds=(0, 1))
    assert d.value == approx(.7)
    with pytest.raises(NotImplementedError):
        d.gradient


def test_l0():
    d = distances.L0(
        np.array([0, .5]),
        np.array([.7, .5]),
        bounds=(0, 1))
    assert d.value == approx(1.)
    with pytest.raises(NotImplementedError):
        d.gradient


@pytest.mark.parametrize('Distance', [
    distances.MeanSquaredDistance,
    distances.MeanAbsoluteDistance,
    distances.Linfinity,
    distances.L0,
])
def test_str_repr(Distance):
    """Tests that str and repr contain the value
    and that str does not fail when initialized
    with a value rather than calculated."""
    reference = np.zeros((10, 10))
    other = np.ones((10, 10))
    d = Distance(reference, other, bounds=(0, 1))
    assert isinstance(str(d), str)
    if 'L0' in str(d):
        assert '100' in str(d)
        assert '100' in repr(d)
    else:
        assert '1.00e+' in str(d)
        assert '1.00e+' in repr(d)
