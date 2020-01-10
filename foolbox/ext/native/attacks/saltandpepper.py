import eagerpy as ep

from ..utils import flatten
from ..utils import atleast_kd


class SaltAndPepperNoiseAttack:
    """
    Parameters
    ----------
    channel_axis : int or None
        If None, all elements of the input will be perturbed indepdently.
    """

    def __init__(self, model, channel_axis):
        self.model = model
        self.channel_axis = channel_axis

    def __call__(self, inputs, labels, *, criterion, steps=1000):
        originals = ep.astensor(inputs)
        labels = ep.astensor(labels)

        def is_adversarial(p: ep.Tensor) -> ep.Tensor:
            """For each input in x, returns true if it is an adversarial for
            the given model and criterion"""
            logits = ep.astensor(self.model.forward(p.tensor))
            return criterion(originals, labels, p, logits)

        x0 = ep.astensor(inputs)

        N = len(x0)
        shape = list(x0.shape)
        if self.channel_axis is not None:
            shape[self.channel_axis] = 1

        min_, max_ = self.model.bounds()
        r = max_ - min_

        result = x0
        is_adv = is_adversarial(result)
        best_advs_norms = ep.where(is_adv, ep.zeros(x0, N), ep.full(x0, N, ep.inf))
        min_probability = ep.zeros(x0, N)
        max_probability = ep.ones(x0, N)
        stepsizes = max_probability / steps
        p = stepsizes

        for step in range(steps):
            # add salt and pepper
            u = ep.uniform(x0, shape)
            p_ = atleast_kd(p, x0.ndim)
            salt = (u >= 1 - p_ / 2).astype(x0.dtype) * r
            pepper = -(u < p_ / 2).astype(x0.dtype) * r
            x = x0 + salt + pepper
            x = ep.clip(x, min_, max_)

            # check if we found new best adversarials
            norms = flatten(x).square().sum(axis=-1).sqrt()
            closer = norms < best_advs_norms
            is_adv = is_adversarial(x)  # TODO: ignore those that are not closer anyway
            is_best_adv = ep.logical_and(is_adv, closer)

            # update results and search space
            result = ep.where(atleast_kd(is_best_adv, x.ndim), x, result)
            best_advs_norms = ep.where(is_best_adv, norms, best_advs_norms)
            min_probability = ep.where(is_best_adv, 0.5 * p, min_probability)
            # we set max_probability a bit higher than p because the relationship
            # between p and norms is not strictly monotonic
            max_probability = ep.where(
                is_best_adv, ep.minimum(p * 1.2, 1.0), max_probability
            )
            remaining = steps - step
            stepsizes = ep.where(
                is_best_adv, (max_probability - min_probability) / remaining, stepsizes
            )
            reset = p == max_probability
            p = ep.where(ep.logical_or(is_best_adv, reset), min_probability, p)
            p = ep.minimum(p + stepsizes, max_probability)

        return result.tensor
