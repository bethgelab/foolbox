import scipy.optimize as so

from .base import Attack
from .base import call_decorator
from .. import nprng


class SLSQPAttack(Attack):
    """Uses SLSQP to minimize the distance between the input and the
    adversarial under the constraint that the input is adversarial."""

    # TODO: add support for criteria that are differentiable (if the network
    # is differentiable) and use this to provide constraint gradients

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True):
        """Uses SLSQP to minimize the distance between the input and the
        adversarial under the constraint that the input is adversarial.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified input. If it is a
            numpy array, label must be passed as well. If it is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original input. Must be passed
            if input is a numpy array, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        x = a.unperturbed
        dtype = a.unperturbed.dtype
        min_, max_ = a.bounds()

        # flatten the input (and remember the shape)
        shape = x.shape
        n = x.size
        x = x.flatten()

        x0 = nprng.uniform(min_, max_, size=x.shape)
        bounds = [(min_, max_)] * n
        options = {'maxiter': 500}

        def fun(x, *args):
            """Objective function with derivative"""
            distance = a.normalized_distance(x.reshape(shape))
            return distance.value, distance.gradient.reshape(-1)

        def eq_constraint(x, *args):
            """Equality constraint"""
            _, is_adv = a.forward_one(x.reshape(shape).astype(dtype))
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

        a.forward_one(result.x.reshape(shape).astype(dtype))
