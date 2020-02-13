from typing import Optional

from .basic_iterative_method import LinfBasicIterativeAttack


class ProjectedGradientDescentAttack(LinfBasicIterativeAttack):
    """Projected Gradient Descent Attack

    Args:
        rel_stepsize: Stepsize relative to epsilon (defaults to 0.01 / 0.3)
        abs_stepsize: If given, it takes precedence over rel_stepsize
    """

    def __init__(
        self,
        rel_stepsize: float = 0.01 / 0.3,
        abs_stepsize: Optional[float] = None,
        steps: int = 40,
        random_start: bool = True,
    ) -> None:
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )
