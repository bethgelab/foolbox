import numpy as np
import pytest
import foolbox.ext.native as fbn
import eagerpy as ep

distances = {0: fbn.l0, 1: fbn.l1, 2: fbn.l2, ep.inf: fbn.linf}

data = {}


def register(f):
    data[f.__name__] = f
    return f


@register
def example_4d(dummy):
    reference = ep.full(dummy, (10, 3, 32, 32), 0.2)
    perturbed = reference + 0.6
    return reference, perturbed


@register
def example_batch(dummy):
    x = ep.arange(dummy, 6).float32().reshape((2, 3))
    x = x / x.max()
    reference = x
    perturbed = 1 - x
    return reference, perturbed


@pytest.fixture(scope="session", params=list(data.keys()))
def reference_perturbed(request, dummy):
    return data[request.param](dummy)


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
def test_distance(reference_perturbed, p):
    reference, perturbed = reference_perturbed

    actual = distances[p](reference, perturbed).numpy()

    diff = perturbed.numpy() - reference.numpy()
    diff = diff.reshape((len(diff), -1))
    desired = np.linalg.norm(diff, ord=p, axis=-1)

    np.testing.assert_allclose(actual, desired, rtol=1e-5)


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
def test_distance_repr_str(p):
    assert str(p) in repr(distances[p])
    assert str(p) in str(distances[p])
