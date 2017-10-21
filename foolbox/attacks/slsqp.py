import numpy as np
import scipy.optimize as so

from .base import Attack


class SLSQPAttack(Attack):
    """Uses SLSQP to minimize the distance between the image and the adversarial
    under the constraint that the image is adversarial."""

    # TODO: add support for criteria that are differentiable (if the network
    # is differentiable) and use this to provide constraint gradients

    def __init__(self, *args, **kwargs):
        super(SLSQPAttack, self).__init__(*args, **kwargs)
        self.last_result = None

    def _apply(self, a):
        image = a.original_image
        dtype = a.original_image.dtype
        min_, max_ = a.bounds()

        # flatten the image (and remember the shape)
        shape = image.shape
        n = np.prod(shape)
        image = image.flatten()

        np.random.seed(42)
        x0 = np.random.uniform(min_, max_, size=image.shape)
        bounds = [(min_, max_)] * n
        options = {'maxiter': 500}

        def fun(x, *args):
            """Objective function with derivative"""
            distance = a.normalized_distance(x.reshape(shape))
            return distance.value, distance.gradient.reshape(-1)

        def eq_constraint(x, *args):
            """Equality constraint"""
            _, is_adv = a.predictions(x.reshape(shape).astype(dtype))
            if is_adv:
                return 0.
            else:
                return 1.

        constraints = [
            {
                'type': 'eq',
                'fun': eq_constraint,
            }
        ]

        result = so.minimize(
            fun,
            x0,
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=constraints,
            options=options)

        # for debugging
        # TODO: store in the Adversarial instance
        self.last_result = result

        a.predictions(result.x.reshape(shape).astype(dtype))
