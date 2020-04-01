from typing import Union, Optional, Any
import numpy as np
import eagerpy as ep

from ..devutils import atleast_kd

from ..models import Model

from ..criteria import Criterion

from ..distances import l0

from .base import MinimizationAttack
from .base import T
from .base import get_is_adversarial
from .base import get_criterion
from .base import get_channel_axis
from .base import raise_if_kwargs


class SinglePixelAttack(MinimizationAttack):
    """Perturbs just a single pixel and sets it to the min or max.

    Args:
        steps : Number of pixels to try out.
        channel_axis : Index of the channel axis in the input data.
    """

    def __init__(self, *, steps: int = 1000, channel_axis: Optional[int] = None):
        self.steps = steps
        self.channel_axis = channel_axis

    distance = l0

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if x.ndim != 4:
            raise NotImplementedError(
                "only implemented for inputs with two spatial dimensions (and one channel and one batch dimension)"
            )

        if self.channel_axis is None:
            channel_axis = get_channel_axis(model, x.ndim)
        else:
            channel_axis = self.channel_axis % x.ndim

        if channel_axis is None:
            raise ValueError(
                "cannot infer the data_format from the model, please specify"
                " channel_axis when initializing the attack"
            )

        if channel_axis == 1:
            h, w = x.shape[2:4]
        elif channel_axis == 3:
            h, w = x.shape[1:3]
        else:
            raise ValueError("expected 'channel_axis' to be 1 or 3, got {channel_axis}")

        min_, max_ = model.bounds
        N = len(x)

        x0 = x

        result = x0
        found = is_adversarial(x0)

        pixels = np.array(
            [np.random.permutation(h * w)[: self.steps] for _ in range(N)]
        )

        for i in range(self.steps):
            px_x = pixels[:, i] % w
            px_y = pixels[:, i] // w

            for value in (max_ - min_, min_ - max_):
                # TODO: replace this with ep.index_update(x0, location, value)
                #  as soon as index_update supports (range, tensor, slice)
                #  location = [range(N), px_x, px_y]
                #  location.insert(channel_axis, slice(None))
                #  location = tuple(location)
                #  x_perturbed = ep.index_update(x0, location, value)

                mask_x = ep.onehot_like(
                    ep.zeros(x, (N, w)), ep.astensor(px_x), value=1
                ).reshape((N, w, 1))
                mask_y = ep.onehot_like(
                    ep.zeros(x, (N, h)), ep.astensor(px_y), value=1
                ).reshape((N, 1, h))
                mask = ep.expand_dims(mask_x * mask_y, channel_axis)

                x_perturbed = ep.clip(x0 + 2 * value * mask, min_, max_)

                is_adv = is_adversarial(x_perturbed)
                found = ep.logical_or(found, is_adv)
                result = ep.where(atleast_kd(is_adv, x0.ndim), x_perturbed, result)

            if found.all():
                break

        return restore_type(result)
