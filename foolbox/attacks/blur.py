from typing import Union, Optional, Any
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import eagerpy as ep

from ..devutils import atleast_kd

from ..models import Model

from ..criteria import Criterion

from ..distances import Distance

from .base import FlexibleDistanceMinimizationAttack
from .base import T
from .base import get_is_adversarial
from .base import get_criterion
from .base import get_channel_axis
from .base import raise_if_kwargs


class GaussianBlurAttack(FlexibleDistanceMinimizationAttack):
    """Blurs the inputs using a Gaussian filter with linearly
    increasing standard deviation.

    Args:
        steps : Number of sigma values tested between 0 and max_sigma.
        channel_axis : Index of the channel axis in the input data.
        max_sigma : Maximally allowed sigma value of the Gaussian blur.
    """

    def __init__(
        self,
        *,
        distance: Optional[Distance] = None,
        steps: int = 1000,
        channel_axis: Optional[int] = None,
        max_sigma: Optional[float] = None,
    ):
        super().__init__(distance=distance)
        self.steps = steps
        self.channel_axis = channel_axis
        self.max_sigma = max_sigma

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

        max_sigma: float
        if self.max_sigma is None:
            if channel_axis == 1:
                h, w = x.shape[2:4]
            elif channel_axis == 3:
                h, w = x.shape[1:3]
            else:
                raise ValueError(
                    "expected 'channel_axis' to be 1 or 3, got {channel_axis}"
                )
            max_sigma = max(h, w)
        else:
            max_sigma = self.max_sigma

        min_, max_ = model.bounds

        x0 = x
        x0_ = x0.numpy()

        result = x0
        found = is_adversarial(x0)

        epsilon = 0.0
        stepsize = 1.0 / self.steps
        for _ in range(self.steps):
            # TODO: reduce the batch size to the ones that haven't been sucessful

            epsilon += stepsize

            sigmas = [epsilon * max_sigma] * x0.ndim
            sigmas[0] = 0
            sigmas[channel_axis] = 0

            # TODO: once we can implement gaussian_filter in eagerpy, avoid converting from numpy
            x_ = gaussian_filter(x0_, sigmas)
            x_ = np.clip(x_, min_, max_)
            x = ep.from_numpy(x0, x_)

            is_adv = is_adversarial(x)
            new_adv = ep.logical_and(is_adv, found.logical_not())
            result = ep.where(atleast_kd(new_adv, x.ndim), x, result)
            found = ep.logical_or(new_adv, found)

            if found.all():
                break

        return restore_type(result)
