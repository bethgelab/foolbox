from ..types import L2, Linf

from .basic_iterative_method import L2BasicIterativeAttack
from .basic_iterative_method import LinfBasicIterativeAttack


class L2FastGradientAttack(L2BasicIterativeAttack):
    """L2 Fast Gradient Method (FGM)"""

    def __init__(self, epsilon: L2):
        super().__init__(epsilon=epsilon, stepsize=epsilon, steps=1)


class LinfFastGradientAttack(LinfBasicIterativeAttack):
    """Fast Gradient Sign Method (FGSM)"""

    def __init__(self, epsilon: Linf):
        super().__init__(epsilon=epsilon, stepsize=epsilon, steps=1)
