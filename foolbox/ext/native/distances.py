import eagerpy as ep
from .devutils import flatten
from .devutils import wrap


# all distances should
# * accept native tensors and EagerPy tensors -> use wrap
# * return the same format as the input -> use restore
# * expect a batch dimension and arbitrary other dimensions -> use flatten
# * accept two arguments, reference and perturbed
# * accept an optional keyword argument bounds


def _normalize(x, bounds):
    if bounds is None:
        return x
    min_, max_ = bounds
    assert x.min() >= min_
    assert x.max() <= max_
    x = (x - min_) / (max_ - min_)
    return x


def _create_lx(norm):
    def lx(reference, perturbed, bounds=None):
        reference, perturbed, restore = wrap(reference, perturbed)
        reference = _normalize(reference, bounds)
        perturbed = _normalize(perturbed, bounds)
        norms = norm(flatten(perturbed - reference), axis=-1)
        return restore(norms)

    lx.__name__ = norm.__name__
    lx.__qualname__ = norm.__name__
    return lx


l0 = _create_lx(ep.norms.l0)
l1 = _create_lx(ep.norms.l1)
l2 = _create_lx(ep.norms.l2)
linf = _create_lx(ep.norms.linf)
lp = _create_lx(ep.norms.lp)
