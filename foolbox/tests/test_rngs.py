import pytest
import random
import numpy as np

from foolbox import rng
from foolbox import nprng


@pytest.mark.parametrize('rng', [rng, nprng])
def test_rng(rng):
    random.seed(66)
    np.random.seed(77)
    x1 = rng.sample()
    random.seed(66)
    np.random.seed(77)
    x2 = rng.sample()
    assert x1 != x2
