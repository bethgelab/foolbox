import eagerpy as ep
import numpy as np


class BinarizationRefinementAttack:
    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        inputs,
        labels,
        *,
        adversarials,
        criterion,
        threshold=None,
        included_in="upper",
    ):
        """For models that preprocess their inputs by binarizing the
        inputs, this attack can improve adversarials found by other
        attacks. It does this by utilizing information about the
        binarization and mapping values to the corresponding value in
        the clean input or to the right side of the threshold.

        Parameters
        ----------
        threshold : float
            The treshold used by the models binarization. If none,
            defaults to (model.bounds()[1] - model.bounds()[0]) / 2.
        included_in : str
            Whether the threshold value itself belongs to the lower or
            upper interval.

        """
        originals = ep.astensor(inputs)
        labels = ep.astensor(labels)

        def is_adversarial(p: ep.Tensor) -> ep.Tensor:
            """For each input in x, returns true if it is an adversarial for
            the given model and criterion"""
            logits = ep.astensor(self.model.forward(p.tensor))
            return criterion(originals, labels, p, logits)

        o = ep.astensor(inputs)
        x = ep.astensor(adversarials)

        min_, max_ = self.model.bounds()
        if threshold is None:
            threshold = (min_ + max_) / 2.0

        assert o.dtype == x.dtype
        dtype = o.dtype

        if dtype == o.backend.float16:
            nptype = np.float16
        elif dtype == o.backend.float32:
            nptype = np.float32
        elif dtype == o.backend.float64:
            nptype = np.float64
        else:
            raise ValueError(
                "expected dtype to be float16, float32 or float64, found '{dtype}'"
            )

        threshold = nptype(threshold)
        offset = nptype(1.0)

        if included_in == "lower":
            lower = threshold
            upper = np.nextafter(threshold, threshold + offset)
        elif included_in == "upper":
            lower = np.nextafter(threshold, threshold - offset)
            upper = threshold
        else:
            raise ValueError(
                "expected included_in to be 'lower' or 'upper', found '{included_in}'"
            )

        assert lower < upper

        p = ep.full_like(o, ep.nan)

        lower = ep.ones_like(o) * lower
        upper = ep.ones_like(o) * upper

        indices = ep.logical_and(o <= lower, x <= lower)
        p = ep.where(indices, o, p)

        indices = ep.logical_and(o <= lower, x >= upper)
        p = ep.where(indices, upper, p)

        indices = ep.logical_and(o >= upper, x <= lower)
        p = ep.where(indices, lower, p)

        indices = ep.logical_and(o >= upper, x >= upper)
        p = ep.where(indices, o, p)

        assert not ep.any(ep.isnan(p))

        is_adv1 = is_adversarial(x)
        is_adv2 = is_adversarial(p)
        assert (
            is_adv1 == is_adv2
        ).all(), "The specified threshold does not match what is done by the model."
        return p.tensor
