from typing import Union, Any, Optional
import eagerpy as ep

from ..criteria import Criterion

from ..models import Model

from .base import FlexibleDistanceMinimizationAttack
from .base import T
from .base import raise_if_kwargs


class InversionAttack(FlexibleDistanceMinimizationAttack):
    """Creates "negative images" by inverting the pixel values. [#Hos16]_

    References:
        .. [#Hos16] Hossein Hosseini, Baicen Xiao, Mayoore Jaiswal, Radha Poovendran,
               "On the Limitation of Convolutional Neural Networks in Recognizing
               Negative Images",
               https://arxiv.org/abs/1607.02533
    """

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, criterion, kwargs

        min_, max_ = model.bounds
        x = min_ + max_ - x
        return restore_type(x)
