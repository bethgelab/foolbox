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


def l0(reference, perturbed, bounds=None):
    reference, perturbed, restore = wrap(reference, perturbed)
    reference = _normalize(reference, bounds)
    perturbed = _normalize(perturbed, bounds)
    norms = ep.norms.l0(flatten(perturbed - reference), axis=-1)
    return restore(norms)


def l1(reference, perturbed, bounds=None):
    reference, perturbed, restore = wrap(reference, perturbed)
    reference = _normalize(reference, bounds)
    perturbed = _normalize(perturbed, bounds)
    norms = ep.norms.l1(flatten(perturbed - reference), axis=-1)
    return restore(norms)


def l2(reference, perturbed, bounds=None):
    reference, perturbed, restore = wrap(reference, perturbed)
    reference = _normalize(reference, bounds)
    perturbed = _normalize(perturbed, bounds)
    norms = ep.norms.l2(flatten(perturbed - reference), axis=-1)
    return restore(norms)


def linf(reference, perturbed, bounds=None):
    reference, perturbed, restore = wrap(reference, perturbed)
    reference = _normalize(reference, bounds)
    perturbed = _normalize(perturbed, bounds)
    norms = ep.norms.linf(flatten(perturbed - reference), axis=-1)
    return restore(norms)


def lp(reference, perturbed, bounds=None):
    reference, perturbed, restore = wrap(reference, perturbed)
    reference = _normalize(reference, bounds)
    perturbed = _normalize(perturbed, bounds)
    norms = ep.norms.lp(flatten(perturbed - reference), axis=-1)
    return restore(norms)
