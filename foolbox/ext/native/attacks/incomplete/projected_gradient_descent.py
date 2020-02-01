from .basic_iterative_method import LinfinityBasicIterativeAttack


class ProjectedGradientDescentAttack(LinfinityBasicIterativeAttack):
    def __init__(self, epsilon=0.3, stepsize=0.01, steps=40, random_start=True):
        super().__init__(
            epsilon=epsilon, stepsize=stepsize, steps=steps, random_start=random_start
        )
