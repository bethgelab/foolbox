from .basic_iterative_method import LinfinityBasicIterativeAttack


class ProjectedGradientDescentAttack(LinfinityBasicIterativeAttack):
    def __call__(
        self,
        inputs,
        labels,
        *,
        rescale=False,
        epsilon=0.3,
        step_size=0.01,
        num_steps=40,
        random_start=True,
    ):
        return super().__call__(
            inputs,
            labels,
            rescale=rescale,
            epsilon=epsilon,
            step_size=step_size,
            num_steps=num_steps,
            random_start=random_start,
        )
