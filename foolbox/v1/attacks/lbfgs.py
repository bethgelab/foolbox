import logging

import numpy as np
import scipy.optimize as so

from .base import Attack
from .base import call_decorator
from .gradient import GradientAttack
from ...utils import crossentropy as utils_ce
from ... import rng


class LBFGSAttack(Attack):
    """Uses L-BFGS-B to minimize the distance between the input and the adversarial
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

    def __init__(self, *args, **kwargs):
        if "approximate_gradient" in kwargs:
            self._approximate_gradient = kwargs["approximate_gradient"]
            del kwargs["approximate_gradient"]
            super(LBFGSAttack, self).__init__(*args, **kwargs)
        else:
            self._approximate_gradient = False
            super(LBFGSAttack, self).__init__(*args, **kwargs)

    def name(self):
        prefix = "Approximate" if self._approximate_gradient else ""
        return "{}{}".format(prefix, self.__class__.__name__)

    @call_decorator
    def __call__(
        self,
        input_or_adv,
        label=None,
        unpack=True,
        epsilon=1e-5,
        num_random_targets=0,
        maxiter=150,
    ):

        """Uses L-BFGS-B to minimize the distance between the input and the
        adversarial as well as the cross-entropy between the predictions for
        the adversarial and the the one-hot encoded target class.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        epsilon : float
            Epsilon of the binary search.
        num_random_targets : int
            Number of random target classes if no target class is given
            by the criterion.
        maxiter : int
            Maximum number of iterations for L-BFGS-B.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not self._approximate_gradient and not a.has_gradient():
            return

        original_class = a.original_class

        target_class = a.target_class
        if target_class is None:
            if num_random_targets == 0 and self._approximate_gradient:
                num_random_targets = 1

            if num_random_targets == 0:
                gradient_attack = GradientAttack()
                gradient_attack(a)
                adv_img = a.perturbed
                if adv_img is None:  # pragma: no coverage
                    # using GradientAttack did not work,
                    # falling back to random target
                    num_random_targets = 1
                    logging.warning(
                        "Using GradientAttack to determine a target class failed,"
                        " falling back to a random target class"
                    )
                else:
                    logits, _ = a.forward_one(adv_img)
                    target_class = np.argmax(logits)
                    target_classes = [target_class]
                    logging.info(
                        "Determined a target class using the GradientAttack: {}".format(
                            target_class
                        )
                    )

            if num_random_targets > 0:

                # draw num_random_targets random classes all of which are
                # different and not the original class

                num_classes = a.num_classes()
                assert num_random_targets <= num_classes - 1

                # sample one more than necessary
                # remove original class from samples
                # should be more efficient than other approaches, see
                # https://github.com/numpy/numpy/issues/2764
                target_classes = rng.sample(range(num_classes), num_random_targets + 1)
                target_classes = [t for t in target_classes if t != original_class]
                target_classes = target_classes[:num_random_targets]

                str_target_classes = [str(t) for t in target_classes]
                logging.info(
                    "Random target classes: {}".format(", ".join(str_target_classes))
                )
        else:
            target_classes = [target_class]

        # avoid mixing GradientAttack and LBFGS Attack
        a._reset()

        for i, target_class in enumerate(target_classes):
            self._optimize(a, target_class, epsilon=epsilon, maxiter=maxiter)

            if len(target_classes) > 1:  # pragma: no coverage
                logging.info(
                    "Best adversarial distance after {} target classes: {}".format(
                        i + 1, a.distance
                    )
                )

    def _optimize(self, a, target_class, epsilon, maxiter):
        x0 = a.unperturbed
        min_, max_ = a.bounds()

        # store the shape for later and operate on the flattened input
        shape = x0.shape
        dtype = x0.dtype
        x0 = x0.flatten().astype(np.float64)

        n = len(x0)
        bounds = [(min_, max_)] * n

        if self._approximate_gradient:

            def distance(x):
                d = a.normalized_distance(x.reshape(shape))
                return d.value

            def crossentropy(x):
                # lbfgs with approx grad does not seem to respect the bounds
                # setting strict to False
                logits, _ = a.forward_one(x.reshape(shape), strict=False)
                ce = utils_ce(logits=logits, label=target_class)
                return ce

            def loss(x, c):
                x = x.astype(dtype)
                v1 = distance(x)
                v2 = crossentropy(x)
                return np.float64(v1 + c * v2)

        else:

            def distance(x):
                d = a.normalized_distance(x.reshape(shape))
                return d.value, d.gradient.reshape(-1)

            def crossentropy(x):
                logits, gradient, _ = a.forward_and_gradient_one(
                    x.reshape(shape), target_class, strict=False
                )
                gradient = gradient.reshape(-1)
                ce = utils_ce(logits=logits, label=target_class)
                return ce, gradient

            def loss(x, c):
                x = x.astype(dtype)
                v1, g1 = distance(x)
                v2, g2 = crossentropy(x)
                v = v1 + c * v2
                g = g1 + c * g2

                a = 1e10
                return np.float64(a * v), np.float64(a * g)

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
                epsilon=approx_grad_eps,
            )

            logging.info(d)

            # LBFGS-B does not always exactly respect the boundaries
            if np.amax(x) > max_ or np.amin(x) < min_:  # pragma: no coverage
                logging.info(
                    "Input out of bounds (min, max = {}, {}). Performing manual clip.".format(
                        np.amin(x), np.amax(x)
                    )
                )
                x = np.clip(x, min_, max_)

            _, is_adversarial = a.forward_one(x.reshape(shape).astype(dtype))
            return is_adversarial

        # finding initial c
        c = epsilon
        for i in range(30):
            c = 2 * c
            is_adversarial = lbfgsb(c)
            logging.info(
                "Tested c = {:.4e}: {}".format(
                    c, ("adversarial" if is_adversarial else "not adversarial")
                )
            )
            if is_adversarial:
                break
        else:  # pragma: no cover
            logging.info(
                "Could not find an adversarial; maybe the model returns wrong gradients"
            )
            return

        # binary search
        c_low = 0
        c_high = c
        while c_high - c_low >= epsilon:
            c_half = (c_low + c_high) / 2
            is_adversarial = lbfgsb(c_half)
            logging.info(
                "Tested c = {:.4e}: {} ({:.4e}, {:.4e})".format(
                    c_half,
                    ("adversarial" if is_adversarial else "not adversarial"),
                    c_low,
                    c_high,
                )
            )
            if is_adversarial:
                c_high = c_half
            else:
                c_low = c_half


class ApproximateLBFGSAttack(LBFGSAttack):
    """Same as :class:`LBFGSAttack` with approximate_gradient set to True.

    """

    def __init__(self, *args, **kwargs):
        assert "approximate_gradient" not in kwargs
        kwargs["approximate_gradient"] = True
        super(ApproximateLBFGSAttack, self).__init__(*args, **kwargs)
