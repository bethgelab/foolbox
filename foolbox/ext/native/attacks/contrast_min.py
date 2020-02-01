import eagerpy as ep

from ..devutils import atleast_kd

from ..models import Model

from ..criteria import Criterion
from ..criteria import misclassification

from .base import MinimizationAttack
from .base import T
from .base import get_is_adversarial


class BinarySearchContrastReductionAttack(MinimizationAttack):
    """Reduces the contrast of the input using a binary search to find the
    smallest adversarial perturbation"""

    def __init__(self, binary_search_steps: int = 15, target: float = 0.5) -> None:
        self.binary_search_steps = binary_search_steps
        self.target = target

    def __call__(
        self,
        model: Model,
        inputs: T,
        labels: T,
        *,
        criterion: Criterion = misclassification,
    ) -> T:

        (x, y), restore_type = ep.astensors_(inputs, labels)
        del inputs, labels

        is_adversarial = get_is_adversarial(criterion, x, y, model)

        min_, max_ = model.bounds
        target = min_ + self.target * (max_ - min_)
        direction = target - x

        lower_bound = ep.zeros(x, len(x))
        upper_bound = ep.ones(x, len(x))
        epsilons = lower_bound
        for _ in range(self.binary_search_steps):
            eps = atleast_kd(epsilons, x.ndim)
            is_adv = is_adversarial(x + eps * direction)
            lower_bound = ep.where(is_adv, lower_bound, epsilons)
            upper_bound = ep.where(is_adv, epsilons, upper_bound)
            epsilons = (lower_bound + upper_bound) / 2

        epsilons = upper_bound
        eps = atleast_kd(epsilons, x.ndim)
        x = x + eps * direction
        return restore_type(x)
