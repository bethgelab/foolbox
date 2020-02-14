"""Internal module with utility functions"""
import eagerpy as ep


def flatten(x: ep.Tensor, keep: int = 1) -> ep.Tensor:
    return x.flatten(start=keep)


def atleast_kd(x: ep.Tensor, k: int) -> ep.Tensor:
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)
