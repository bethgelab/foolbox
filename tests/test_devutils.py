import pytest
import foolbox.ext.native as fbn
import eagerpy as ep

distances = {0: fbn.l0, 1: fbn.l1, 2: fbn.l2, ep.inf: fbn.linf}

data = {}


def register(f):
    data[f.__name__] = f
    return f


@register
def example_0_arg(dummy):
    return ()


@register
def example_1_arg(dummy):
    return (ep.full(dummy, (10, 3, 32, 32), 0.2),)


@register
def example_2_args(dummy):
    return ep.full(dummy, (10, 3, 32, 32), 0.2), ep.zeros(dummy, 7)


@register
def example_3_args(dummy):
    return (
        ep.full(dummy, (10, 3, 32, 32), 0.2),
        ep.zeros(dummy, 7),
        ep.ones(dummy, (5, 5)),
    )


@pytest.fixture(scope="session", params=list(data.keys()))
def args(request, dummy):
    return data[request.param](dummy)


def transparent_wrap(*args):
    *args, restore = fbn.devutils.wrap(*args)
    for arg in args:
        assert ep.istensor(arg)
    return restore(*args)


def transparent_unwrap(*args):
    *args, restore = fbn.devutils.unwrap(*args)
    for arg in args:
        assert not ep.istensor(arg)
        assert ep.istensor(ep.astensor(arg))
    return restore(*args)


@pytest.mark.parametrize("transparent_function", [transparent_unwrap, transparent_wrap])
@pytest.mark.parametrize("unwrap", [False, True])
def test_transparent(args, transparent_function, unwrap):
    assert isinstance(args, tuple)
    if unwrap:
        args = tuple(x.tensor for x in args)
    for x in args:
        assert isinstance(x, type(args[0]))
    results = transparent_function(*args)
    if len(args) == 1:
        results = (results,)
    assert isinstance(results, tuple)
    for x in results:
        assert isinstance(x, type(args[0]))


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_atleast_kd_1d(dummy, k):
    x = ep.zeros(dummy, (10,))
    x = fbn.devutils.atleast_kd(x, k)
    assert x.shape[0] == 10
    assert x.ndim == k


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_atleast_kd_3d(dummy, k):
    x = ep.zeros(dummy, (10, 5, 3))
    x = fbn.devutils.atleast_kd(x, k)
    assert x.shape[:3] == (10, 5, 3)
    assert x.ndim == max(k, 3)


def test_flatten_2d(dummy):
    x = ep.zeros(dummy, (4, 5))
    x = fbn.devutils.flatten(x)
    assert x.shape == (4, 5)


def test_flatten_3d(dummy):
    x = ep.zeros(dummy, (4, 5, 6))
    x = fbn.devutils.flatten(x)
    assert x.shape == (4, 30)


def test_flatten_4d(dummy):
    x = ep.zeros(dummy, (4, 5, 6, 7))
    x = fbn.devutils.flatten(x)
    assert x.shape == (4, 210)
