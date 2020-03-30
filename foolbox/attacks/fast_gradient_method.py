from .gradient_descent_base import L1BaseGradientDescent
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent


class L1FastGradientAttack(L1BaseGradientDescent):
    """Fast Gradient Method (FGM) using the L1 norm

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, random_start: bool = False):
        super().__init__(
            rel_stepsize=1.0, steps=1, random_start=random_start,
        )


class L2FastGradientAttack(L2BaseGradientDescent):
    """Fast Gradient Method (FGM)

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, random_start: bool = False):
        super().__init__(
            rel_stepsize=1.0, steps=1, random_start=random_start,
        )


class LinfFastGradientAttack(LinfBaseGradientDescent):
    """Fast Gradient Sign Method (FGSM)

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, random_start: bool = False):
        super().__init__(
            rel_stepsize=1.0, steps=1, random_start=random_start,
        )
