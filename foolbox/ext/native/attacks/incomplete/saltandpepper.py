import eagerpy as ep
from typing import Optional

from ..criteria import misclassification

from ..devutils import flatten
from ..devutils import atleast_kd
from ..devutils import wrap

from .base import MinimizationAttack
from .base import get_is_adversarial

from ..models.base import Model


class SaltAndPepperNoiseAttack(MinimizationAttack):
    """Increases the amount of salt and pepper noise until the input is misclassified.

    Parameters
    ----------
    steps
        The number of steps to run
    across_channels
        Whether the noise should be the same across all channels
    """

    def __init__(self, steps: int = 1000, across_channels: bool = True):
        self.steps = steps
        self.across_channels = across_channels

    def __call__(
        self,
        model: Model,
        inputs,
        labels,
        *,
        criterion=misclassification,
        channel_axis: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        channel_axis
            The axis across which the noise should be the same (if across_channels is True).
            If None, will be automatically inferred from the model if possible.
        """
        inputs, labels, restore = wrap(inputs, labels)
        is_adversarial = get_is_adversarial(criterion, inputs, labels, model)

        x0 = inputs
        N = len(x0)
        shape = list(x0.shape)
        if self.across_channels and x0.ndim > 2:
            if channel_axis is None and not hasattr(model, "data_format"):
                raise ValueError(
                    "cannot infer the data_format from the model, please specify"
                    " channel_axis when calling the attack"
                )
            elif channel_axis is None:
                data_format = model.data_format  # type: ignore
                if (
                    data_format is None
                    or data_format != "channels_first"
                    and data_format != "channels_last"
                ):
                    raise ValueError(
                        f"expected data_format to be 'channels_first' or 'channels_last'"
                    )
                channel_axis = 1 if data_format == "channels_first" else x0.ndim - 1
            elif not 0 <= channel_axis < x0.ndim:
                raise ValueError(f"expected channel_axis to be in [0, {x0.ndim})")

            shape[channel_axis] = 1

        min_, max_ = model.bounds()
        r = max_ - min_

        result = x0
        is_adv = is_adversarial(result)
        best_advs_norms = ep.where(is_adv, ep.zeros(x0, N), ep.full(x0, N, ep.inf))
        min_probability = ep.zeros(x0, N)
        max_probability = ep.ones(x0, N)
        stepsizes = max_probability / self.steps
        p = stepsizes

        for step in range(self.steps):
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
            remaining = self.steps - step
            stepsizes = ep.where(
                is_best_adv, (max_probability - min_probability) / remaining, stepsizes
            )
            reset = p == max_probability
            p = ep.where(ep.logical_or(is_best_adv, reset), min_probability, p)
            p = ep.minimum(p + stepsizes, max_probability)

        return restore(result)
