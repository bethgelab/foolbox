from typing import Union, Optional, Any
import eagerpy as ep

from ..criteria import Criterion

from .base import Model
from .base import T
from .base import get_is_adversarial
from .base import get_criterion
from .base import Attack
from .spatial_attack_transformations import rotate_and_shift


class SpatialAttack(Attack):
    """Adversarially chosen rotations and translations [1].
    This implementation is based on the reference implementation by
    Madry et al.: https://github.com/MadryLab/adversarial_spatial
    References
    ----------
    .. [1] Logan Engstrom*, Brandon Tran*, Dimitris Tsipras*,
           Ludwig Schmidt, Aleksander Mądry: "A Rotation and a
           Translation Suffice: Fooling CNNs with Simple Transformations",
           http://arxiv.org/abs/1712.02779
    """
    def __init__(
            self,
            max_translation: float = 3,
            max_rotation: int = 30,
            channel_axis: float = 1,

    ):
        self.max_translation = max_translation
        self.max_rotation = max_rotation

    def __call__(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Criterion, T],
    ):
        x, restore_type = ep.astensor_(inputs)
        del inputs
        criterion = get_criterion(criterion)

        is_adversarial = get_is_adversarial(criterion, model)
        print(is_adversarial(x))

        if x.ndim != 4:
            raise NotImplementedError(
                "only implemented for inputs with two spatial dimensions (and one channel and one batch dimension)"
            )
        print('here')

        xp = self.run(model, x, criterion, restore_type)
        success = is_adversarial(xp)

        xp_ = restore_type(xp)
        return xp_, success

    def run(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Criterion, T],
            restore_type,
            *,
            early_stop: Optional[float] = None,
            **kwargs: Any,
    ) -> T:
        is_adversarial = get_is_adversarial(criterion, model)

        # TODO @lukas: search from origin 0, 1, -1, 2, ... for rot, and trans
        # TODO @lukas: mask s.t. adversarials are not further transformed
        for rot in range(self.max_rotation):
            for trans_x in range(self.max_translation):
                for trans_y in range(self.max_translation):
                    x_p = rotate_and_shift(inputs, restore_type,
                                           translation=(trans_x, trans_y),
                                           rotation=rot)
                    is_adversarial = get_is_adversarial(criterion, model)
                    if ep.all(is_adversarial(x_p)):
                        print('all misclassified', is_adversarial(x_p))
                        return x_p
        return x_p

    def repeat(self, times: int) -> "Attack":
        return self

