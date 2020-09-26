from .gradient_descent_base import L1BaseGradientDescent
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent
from ..models.base import Model
from ..criteria import Misclassification, TargetedMisclassification
from .base import T
from typing import Union, Any


class L1FastGradientAttack(L1BaseGradientDescent):
    """Fast Gradient Method (FGM) using the L1 norm

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, random_start: bool = False):
        super().__init__(
            rel_stepsize=1.0, steps=1, random_start=random_start,
        )

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        if hasattr(criterion, "target_classes"):
            raise ValueError("unsupported criterion")

        return super().run(
            model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs
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

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        if hasattr(criterion, "target_classes"):
            raise ValueError("unsupported criterion")

        return super().run(
            model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs
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

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        if hasattr(criterion, "target_classes"):
            raise ValueError("unsupported criterion")

        return super().run(
            model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs
        )
