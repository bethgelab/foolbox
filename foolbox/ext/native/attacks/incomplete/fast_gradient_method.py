from .basic_iterative_method import L2BasicIterativeAttack
from .basic_iterative_method import LinfinityBasicIterativeAttack


class L2FastGradientAttack(L2BasicIterativeAttack):
    """L2 Fast Gradient Method (FGM)"""

    def __init__(self, epsilon):
        super().__init__(epsilon=epsilon, stepsize=epsilon, steps=1)


class LinfinityFastGradientAttack(LinfinityBasicIterativeAttack):
    """Fast Gradient Sign Method (FGSM)"""

    def __init__(self, epsilon):
        super().__init__(epsilon=epsilon, stepsize=epsilon, steps=1)
