from .basic_iterative_method import L2BasicIterativeAttack
from .basic_iterative_method import LinfBasicIterativeAttack


class L2FastGradientAttack(L2BasicIterativeAttack):
    """L2 Fast Gradient Method (FGM)"""

    def __init__(self) -> None:
        super().__init__(rel_stepsize=1.0, steps=1)


class LinfFastGradientAttack(LinfBasicIterativeAttack):
    """Fast Gradient Sign Method (FGSM)"""

    def __init__(self) -> None:
        super().__init__(rel_stepsize=1.0, steps=1)
