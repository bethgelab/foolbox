import numpy as np
import warnings
import logging

from .base import Attack
from .base import call_decorator


class BinarizationRefinementAttack(Attack):
    """For models that preprocess their inputs by binarizing the
    inputs, this attack can improve adversarials found by other
    attacks. It does os by utilizing information about the
    binarization and mapping values to the corresponding value in
    the clean input or to the right side of the threshold.

    """

    @call_decorator
    def __call__(
        self,
        input_or_adv,
        label=None,
        unpack=True,
        starting_point=None,
        threshold=None,
        included_in="upper",
    ):

        """For models that preprocess their inputs by binarizing the
        inputs, this attack can improve adversarials found by other
        attacks. It does os by utilizing information about the
        binarization and mapping values to the corresponding value in
        the clean input or to the right side of the threshold.

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
        starting_point : `numpy.ndarray`
            Adversarial input to use as a starting point.
        threshold : float
            The treshold used by the models binarization. If none,
            defaults to (model.bounds()[1] - model.bounds()[0]) / 2.
        included_in : str
            Whether the threshold value itself belongs to the lower or
            upper interval.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        self._starting_point = starting_point
        self.initialize_starting_point(a)

        if a.perturbed is None:
            warnings.warn(
                "This attack can only be applied to an adversarial"
                " found by another attack, either by calling it with"
                " an Adversarial object or by passing a starting_point"
            )
            return

        assert a.perturbed.dtype == a.unperturbed.dtype
        dtype = a.unperturbed.dtype

        assert np.issubdtype(dtype, np.floating)

        min_, max_ = a.bounds()

        if threshold is None:
            threshold = (min_ + max_) / 2.0

        threshold = dtype.type(threshold)
        offset = dtype.type(1.0)

        if included_in == "lower":
            lower = threshold
            upper = np.nextafter(threshold, threshold + offset)
        elif included_in == "upper":
            lower = np.nextafter(threshold, threshold - offset)
            upper = threshold
        else:
            raise ValueError('included_in must be "lower" or "upper"')

        logging.info(
            "Intervals: [{}, {}] and [{}, {}]".format(min_, lower, upper, max_)
        )

        assert type(lower) == dtype.type
        assert type(upper) == dtype.type

        assert lower < upper

        o = a.unperturbed
        x = a.perturbed

        p = np.full_like(o, np.nan)

        indices = np.logical_and(o <= lower, x <= lower)
        p[indices] = o[indices]

        indices = np.logical_and(o <= lower, x >= upper)
        p[indices] = upper

        indices = np.logical_and(o >= upper, x <= lower)
        p[indices] = lower

        indices = np.logical_and(o >= upper, x >= upper)
        p[indices] = o[indices]

        assert not np.any(np.isnan(p))

        logging.info(
            "distance before the {}: {}".format(self.__class__.__name__, a.distance)
        )
        _, is_adversarial = a.forward_one(p)
        assert is_adversarial, (
            "The specified thresholding does not" " match what is done by the model."
        )
        logging.info(
            "distance after the {}: {}".format(self.__class__.__name__, a.distance)
        )

    def initialize_starting_point(self, a):
        starting_point = self._starting_point

        if a.perturbed is not None:
            if starting_point is not None:  # pragma: no cover
                warnings.warn(
                    "Ignoring starting_point because the attack"
                    " is applied to a previously found adversarial."
                )
            return

        if starting_point is not None:
            a.forward_one(starting_point)
            assert (
                a.perturbed is not None
            ), "Invalid starting point provided. Please provide a starting point that is adversarial."
            return
