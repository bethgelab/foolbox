import pytest
import numpy as np
from foolbox import distances


def test_abstract_distance():
    with pytest.raises(TypeError):
        distances.Distance()


def test_base_distance():

    class TestDistance(distances.Distance):

        def _calculate(self):
            return 22, 2

    distance = TestDistance(None, None)
    assert distance.name() == 'TestDistance'
    assert distance.value() == 22
    assert distance.gradient() == 2
    assert '2.2' in str(distance)
    assert 'TestDistance' in str(distance)
    assert distance == distance
    assert not distance < distance
    assert not distance > distance
    assert distance <= distance
    assert distance >= distance


def test_mse():
    assert distances.MSE == distances.MeanSquaredDistance


def test_mean_squared_distance():
    d = distances.MeanSquaredDistance(np.array([0, 2]), np.array([2, 2]))
    assert d.value() == 2.
    assert (d.gradient() == np.array([2, 0])).all()

    assert str(d)[:5] == 'MSE ='


def test_mean_absolute_distance():
    d = distances.MeanAbsoluteDistance(np.array([0, 2]), np.array([2, 2]))
    assert d.value() == 1.
    assert (d.gradient() == np.array([1, 0])).all()

    assert str(d)[:5] == 'MAE ='


@pytest.mark.parametrize('Distance', [
    distances.MeanSquaredDistance,
    distances.MeanAbsoluteDistance,
])
def test_str_repr(Distance):
    """Tests that str and repr contain the value
    and that str does not fail when initialized
    with a value rather than calculated."""
    d = Distance(value=33)
    assert isinstance(str(d), str)
    assert '3' in str(d)
    assert '3' in repr(d)
    d = Distance(value=np.inf)
    assert isinstance(str(d), str)
