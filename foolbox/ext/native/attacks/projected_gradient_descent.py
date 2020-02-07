from ..types import Linf

from .basic_iterative_method import LinfBasicIterativeAttack


class ProjectedGradientDescentAttack(LinfBasicIterativeAttack):
    def __init__(
        self,
        epsilon: Linf = Linf(0.3),
        stepsize: float = 0.01,
        steps: int = 40,
        random_start: bool = True,
    ) -> None:
        super().__init__(
            epsilon=epsilon, stepsize=stepsize, steps=steps, random_start=random_start
        )
