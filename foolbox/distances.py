from abc import ABC, abstractmethod
from typing import TypeVar
import eagerpy as ep

from .devutils import flatten


T = TypeVar("T")


class Distance(ABC):
    @abstractmethod
    def __call__(self, reference: T, perturbed: T) -> T:
        ...


class LpDistance(Distance):
    def __init__(self, p: float):
        self.p = p

    def __repr__(self) -> str:
        return f"LpDistance({self.p})"

    def __str__(self) -> str:
        return f"L{self.p} distance"

    def __call__(self, reference: T, perturbed: T) -> T:
        """Calculates the distance from reference to perturbed using the Lp norm.

        Parameters
        ----------
        reference : T
            A batch of reference inputs.
        perturbed : T
            A batch of perturbed inputs.

        Returns
        -------
        T
            Returns a batch of distances as a 1D tensor.

        """
        (x, y), restore_type = ep.astensors_(reference, perturbed)
        norms = ep.norms.lp(flatten(y - x), self.p, axis=-1)
        return restore_type(norms)


l0 = LpDistance(0)
l1 = LpDistance(1)
l2 = LpDistance(2)
linf = LpDistance(ep.inf)
