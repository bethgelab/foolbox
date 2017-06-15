import random
import logging

import numpy as np
import scipy.optimize as so

from .base import Attack
from .gradient import GradientAttack

from foolbox import utils


class LBFGSAttack(Attack):
    """Uses L-BFGS-B to minimize the distance between the image and the adversarial
    as well as the cross-entropy between the predictions for the adversarial
    and the the one-hot encoded target class.

    If the criterion does not have a target class, a random class is chosen
    from the set of all classes except the original one.

    Notes
    -----
    This implementation generalizes algorithm 1 in [1]_ to support other
    targeted criteria and other distance measures.

    References
    ----------

    .. [1] https://arxiv.org/abs/1510.05328

    """

    def __init__(self, *args, approximate_gradient=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._approximate_gradient = approximate_gradient

    def name(self):
        prefix = 'Approximate' if self._approximate_gradient else ''
        return '{}{}'.format(prefix, self.__class__.__name__)

    def _apply(
            self,
            a,
            epsilon=1e-5,
            num_random_targets=0,
            maxiter=150,
            verbose=False):

        if not self._approximate_gradient and not a.has_gradient():
            return

        original_class = a.original_class

        target_class = a.target_class()
        if target_class is None:
            if num_random_targets == 0:
                gradient_attack = GradientAttack()
                gradient_attack(a)
                adv_img = a.image
                if adv_img is None:  # pragma: no coverage
                    # using GradientAttack did not work,
                    # falling back to random target
                    num_random_targets = 1
                    logging.info('Using GradientAttack to determine a target class failed, falling back to a random target class')  # noqa: E501
                else:
                    logits, _ = a.predictions(adv_img)
                    target_class = np.argmax(logits)
                    target_classes = [target_class]
                    logging.info('Determined a target class using the GradientAttack: {}'.format(target_class))  # noqa: E501
            else:
                num_random_targets = 1

            if num_random_targets > 0:

                # draw num_random_targets random classes all of which are
                # different and not the original class

                num_classes = a.num_classes()
                assert num_random_targets <= num_classes - 1

                # sample one more than necessary
                # remove original class from samples
                # should be more efficient than other approaches, see
                # https://github.com/numpy/numpy/issues/2764
                target_classes = random.sample(
                    range(num_classes), num_random_targets + 1)
                target_classes = [t for t in target_classes if t != original_class]  # noqa: E501
                target_classes = target_classes[:num_random_targets]

                str_target_classes = [str(t) for t in target_classes]
                logging.info('Random target classes: {}'.format(', '.join(str_target_classes)))  # noqa: E501
        else:
            target_classes = [target_class]

        for i, target_class in enumerate(target_classes):
            self._optimize(
                a, target_class,
                epsilon=epsilon, maxiter=maxiter, verbose=verbose)

            if verbose and len(target_classes) > 1:  # pragma: no coverage
                logging.info('Best adversarial distance after {} target classes: {}'.format(i + 1, a.distance))  # noqa: E501

    def _optimize(self, a, target_class, epsilon, maxiter, verbose):
        image = a.original_image
        min_, max_ = a.bounds()

        # store the shape for later and operate on the flattened image
        shape = image.shape
        # dtype = image.dtype
        image = image.flatten().astype(np.float64)

        n = len(image)
        bounds = [(min_, max_)] * n

        x0 = image

        if self._approximate_gradient:

            def distance(x):
                distance = a.normalized_distance(x.reshape(shape))
                return distance.value()

            def crossentropy(x):
                # lbfgs with approx grad does not seem to respect the bounds
                # setting strict to False
                logits, _ = a.predictions(x.reshape(shape), strict=False)
                ce = utils.crossentropy(logits=logits, label=target_class)
                return ce

            def loss(x, c):
                v1 = distance(x)
                v2 = crossentropy(x)
                return v1 + c * v2

        else:

            def distance(x):
                distance = a.normalized_distance(x.reshape(shape))
                return distance.value(), distance.gradient().reshape(-1)

            def crossentropy(x):
                logits, gradient, _ = a.predictions_and_gradient(
                    x.reshape(shape), target_class, strict=False)
                gradient = gradient.reshape(-1)
                ce = utils.crossentropy(logits=logits, label=target_class)
                return ce, gradient

            def loss(x, c):
                v1, g1 = distance(x)
                v2, g2 = crossentropy(x)
                v = v1 + c * v2
                g = g1 + c * g2

                a = 1e10
                return a * v, a * g

        def lbfgsb(c):
            approx_grad_eps = (max_ - min_) / 100
            x, f, d = so.fmin_l_bfgs_b(
                loss,
                x0,
                args=(c,),
                approx_grad=self._approximate_gradient,
                bounds=bounds,
                m=15,
                maxiter=maxiter,
                epsilon=approx_grad_eps)

            logging.info(d)

            _, is_adversarial = a.predictions(x.reshape(shape))
            return is_adversarial

        # finding initial c
        c = epsilon
        for i in range(30):
            c = 2 * c
            is_adversarial = lbfgsb(c)
            if verbose:
                logging.info('Tested c = {:.4e}: {}'.format(
                    c,
                    ('adversarial' if is_adversarial else 'not adversarial')))
            if is_adversarial:
                break
        else:  # pragma: no cover
            if verbose:
                logging.info('Could not find an adversarial; maybe the model returns wrong gradients')  # noqa: E501
            return

        # binary search
        c_low = 0
        c_high = c
        while c_high - c_low >= epsilon:
            c_half = (c_low + c_high) / 2
            is_adversarial = lbfgsb(c_half)
            if verbose:
                logging.info('Tested c = {:.4e}: {} ({:.4e}, {:.4e})'.format(
                    c_half,
                    ('adversarial' if is_adversarial else 'not adversarial'),
                    c_low,
                    c_high))
            if is_adversarial:
                c_high = c_half
            else:
                c_low = c_half


class ApproximateLBFGSAttack(LBFGSAttack):
    """Same as :class:`LBFGSBAttack` with approximate_gradient set to True.

    """

    def __init__(self, *args, **kwargs):
        assert 'approximate_gradient' not in kwargs
        super().__init__(*args, approximate_gradient=True, **kwargs)
