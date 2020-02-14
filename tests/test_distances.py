from typing import Tuple, Any, Dict, Callable, TypeVar
import numpy as np
import pytest
import foolbox as fbn
import eagerpy as ep

distances = {
    0: fbn.distances.l0,
    1: fbn.distances.l1,
    2: fbn.distances.l2,
    ep.inf: fbn.distances.linf,
}

data: Dict[str, Callable[..., Tuple[ep.Tensor, ep.Tensor]]] = {}

FuncType = Callable[..., Tuple[ep.Tensor, ep.Tensor]]
F = TypeVar("F", bound=FuncType)


def register(f: F) -> F:
    data[f.__name__] = f
    return f


@register
def example_4d(dummy: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:
    reference = ep.full(dummy, (10, 3, 32, 32), 0.2)
    perturbed = reference + 0.6
    return reference, perturbed


@register
def example_batch(dummy: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:
    x = ep.arange(dummy, 6).float32().reshape((2, 3))
    x = x / x.max()
    reference = x
    perturbed = 1 - x
    return reference, perturbed


@pytest.fixture(scope="session", params=list(data.keys()))
def reference_perturbed(request: Any, dummy: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:
    return data[request.param](dummy)


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
def test_distance(reference_perturbed: Tuple[ep.Tensor, ep.Tensor], p: float) -> None:
    reference, perturbed = reference_perturbed

    actual = distances[p](reference, perturbed).numpy()

    diff = perturbed.numpy() - reference.numpy()
    diff = diff.reshape((len(diff), -1))
    desired = np.linalg.norm(diff, ord=p, axis=-1)

    np.testing.assert_allclose(actual, desired, rtol=1e-5)


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
def test_distance_repr_str(p: float) -> None:
    assert str(p) in repr(distances[p])
    assert str(p) in str(distances[p])


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
def test_distance_clip(
    reference_perturbed: Tuple[ep.Tensor, ep.Tensor], p: float
) -> None:
    reference, perturbed = reference_perturbed

    ds = distances[p](reference, perturbed).numpy()
    epsilon = np.median(ds)
    too_large = ds > epsilon

    desired = np.where(too_large, epsilon, ds)

    perturbed = distances[p].clip_perturbation(reference, perturbed, epsilon)
    actual = distances[p](reference, perturbed).numpy()

    np.testing.assert_allclose(actual, desired)
