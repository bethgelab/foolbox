from typing import Union, Any
import eagerpy as ep

from ..criteria import Criterion

from ..models import Model

from .base import MinimizationAttack
from .base import T


class InversionAttack(MinimizationAttack):
    """Creates "negative images" by inverting the pixel values according to [1]_.

    References
    ----------
    .. [1] Hossein Hosseini, Baicen Xiao, Mayoore Jaiswal, Radha Poovendran,
           "On the Limitation of Convolutional Neural Networks in Recognizing
           Negative Images",
            https://arxiv.org/abs/1607.02533
    """

    def __call__(
        self, model: Model, inputs: T, criterion: Union[Criterion, Any] = None
    ) -> T:
        x, restore_type = ep.astensor_(inputs)
        del inputs, criterion

        min_, max_ = model.bounds
        x = min_ + max_ - x
        return restore_type(x)
