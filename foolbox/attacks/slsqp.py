import scipy.optimize as so

from .base import Attack
from .base import call_decorator
from .. import nprng


class SLSQPAttack(Attack):
    """Uses SLSQP to minimize the distance between the image and the
    adversarial under the constraint that the image is adversarial."""

    # TODO: add support for criteria that are differentiable (if the network
    # is differentiable) and use this to provide constraint gradients

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True):
        """Uses SLSQP to minimize the distance between the image and the
        adversarial under the constraint that the image is adversarial.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified image. If image is a
            numpy array, label must be passed as well. If image is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original image. Must be passed
            if image is a numpy array, must not be passed if image is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial image, otherwise returns
            the Adversarial object.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        image = a.original_image
        dtype = a.original_image.dtype
        min_, max_ = a.bounds()

        # flatten the image (and remember the shape)
        shape = image.shape
        n = image.size
        image = image.flatten()

        x0 = nprng.uniform(min_, max_, size=image.shape)
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

        a.predictions(result.x.reshape(shape).astype(dtype))
