import eagerpy as ep
import functools

from .devutils import flatten
from .devutils import wrap


# all distances should
# * accept native tensors and EagerPy tensors -> use wrap
# * return the same format as the input -> use restore
# * expect a batch dimension and arbitrary other dimensions -> use flatten
# * accept two arguments, reference and perturbed
# * accept an optional keyword argument bounds


def normalize(x, bounds):
    if bounds is None:
        return x
    min_, max_ = bounds
    assert x.min() >= min_
    assert x.max() <= max_
    x = (x - min_) / (max_ - min_)
    return x


def norm_to_distance(norm):
    @functools.wraps(norm)
    def distance(reference, perturbed, bounds=None):
        reference, perturbed, restore = wrap(reference, perturbed)
        reference = normalize(reference, bounds)
        perturbed = normalize(perturbed, bounds)
        norms = norm(flatten(perturbed - reference), axis=-1)
        return restore(norms)

    return distance


l0 = norm_to_distance(ep.norms.l0)
l1 = norm_to_distance(ep.norms.l1)
l2 = norm_to_distance(ep.norms.l2)
linf = norm_to_distance(ep.norms.linf)
