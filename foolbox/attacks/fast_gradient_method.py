from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent


class L2FastGradientAttack(L2BaseGradientDescent):
    """Fast Gradient Method (FGM)

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, random_start: bool = False,) -> None:
        super().__init__(
            rel_stepsize=1.0, steps=1, random_start=random_start,
        )


class LinfFastGradientAttack(LinfBaseGradientDescent):
    """Fast Gradient Sign Method (FGSM)

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, random_start: bool = False,) -> None:
        super().__init__(
            rel_stepsize=1.0, steps=1, random_start=random_start,
        )
