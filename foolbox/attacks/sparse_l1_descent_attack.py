from typing import Optional
import eagerpy as ep
import numpy as np

from ..devutils import flatten
from ..devutils import atleast_kd

from ..types import Bounds

from .gradient_descent_base import L1BaseGradientDescent
from .gradient_descent_base import normalize_lp_norms


class SparseL1DescentAttack(L1BaseGradientDescent):
    """Sparse L1 Descent Attack [#Tra19]_.

    Args:
        rel_stepsize: Stepsize relative to epsilon.
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps.
        random_start : Controls whether to randomly start within allowed epsilon ball.

    References:
        .. [#Tra19] Florian Tramèr, Dan Boneh, "Adversarial Training and
        Robustness for Multiple Perturbations"
        https://arxiv.org/abs/1904.13000
    """

    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        bad_pos = ep.logical_or(
            ep.logical_and(x == bounds.lower, gradients < 0),
            ep.logical_and(x == bounds.upper, gradients > 0),
        )
        gradients = ep.where(bad_pos, ep.zeros_like(gradients), gradients)

        abs_gradients = gradients.abs()
        quantiles = np.quantile(
            flatten(abs_gradients).numpy(), q=self.quantile, axis=-1
        )
        keep = abs_gradients >= atleast_kd(
            ep.from_numpy(gradients, quantiles), gradients.ndim
        )
        e = ep.where(keep, gradients.sign(), ep.zeros_like(gradients))
        return normalize_lp_norms(e, p=1)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        # based on https://github.com/ftramer/MultiRobustness/blob/ad41b63235d13b1b2a177c5f270ab9afa74eee69/pgd_attack.py#L110
        delta = flatten(x - x0)
        norms = delta.norms.l1(axis=-1)
        if (norms <= epsilon).all():
            return x

        n, d = delta.shape
        abs_delta = abs(delta)
        mu = -ep.sort(-abs_delta, axis=-1)
        cumsums = mu.cumsum(axis=-1)
        js = 1.0 / ep.arange(x, 1, d + 1).astype(x.dtype)
        temp = mu - js * (cumsums - epsilon)
        guarantee_first = ep.arange(x, d).astype(x.dtype) / d
        # guarantee_first are small values (< 1) that we add to the boolean
        # tensor (only 0 and 1) to break the ties and always return the first
        # argmin, i.e. the first value where the boolean tensor is 0
        # (otherwise, this is not guaranteed on GPUs, see e.g. PyTorch)
        rho = ep.argmin((temp > 0).astype(x.dtype) + guarantee_first, axis=-1)
        theta = 1.0 / (1 + rho.astype(x.dtype)) * (cumsums[range(n), rho] - epsilon)
        delta = delta.sign() * ep.maximum(abs_delta - theta[..., ep.newaxis], 0)
        delta = delta.reshape(x.shape)
        return x0 + delta

    def __init__(
        self,
        *,
        quantile: float = 0.99,
        rel_stepsize: float = 0.2,
        abs_stepsize: Optional[float] = None,
        steps: int = 10,
        random_start: bool = False,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )
        if not 0 <= quantile <= 1:
            raise ValueError(f"quantile needs to be between 0 and 1, got {quantile}")
        self.quantile = quantile
