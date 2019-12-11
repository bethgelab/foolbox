import eagerpy as ep
from .utils import flatten


def l2(x):
    x = ep.astensor(x)
    norms = flatten(x).square().sum(axis=-1).sqrt()
    return norms.tensor
