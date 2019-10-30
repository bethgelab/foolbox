from .basic_iterative_method import L2BasicIterativeAttack
from .basic_iterative_method import LinfinityBasicIterativeAttack


class L2FastGradientAttack(L2BasicIterativeAttack):
    """L2 Fast Gradient Method (FGM)"""

    def __call__(
        self, inputs, labels, *, rescale=False, epsilon=2.0,
    ):
        return super().__call__(
            inputs,
            labels,
            rescale=rescale,
            epsilon=epsilon,
            step_size=epsilon,
            num_steps=1,
        )


class LinfinityFastGradientAttack(LinfinityBasicIterativeAttack):
    """Fast Gradient Sign Method (FGSM)"""

    def __call__(
        self, inputs, labels, *, rescale=False, epsilon=0.3,
    ):
        return super().__call__(
            inputs,
            labels,
            rescale=rescale,
            epsilon=epsilon,
            step_size=epsilon,
            num_steps=1,
        )
