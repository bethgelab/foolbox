import eagerpy as ep
from abc import ABC, abstractmethod
from typing import overload, Any
from .devutils import wrap


class Criterion(ABC):
    @abstractmethod
    def __repr__(self):
        ...

    @overload
    def __call__(
        self,
        inputs: ep.Tensor,
        labels: ep.Tensor,
        perturbed: ep.Tensor,
        logits: ep.Tensor,
    ) -> ep.Tensor:
        ...

    @overload  # noqa: F811
    def __call__(self, inputs: Any, labels: Any, perturbed: Any, logits: Any) -> Any:
        ...

    @abstractmethod  # noqa: F811
    def __call__(self, inputs, labels, perturbed, logits):
        ...

    def __and__(self, other):
        return _And(self, other)


class _And(Criterion):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __repr__(self):
        return f"{self.a!r} & {self.b!r}"

    def __call__(self, inputs, labels, perturbed, logits):
        inputs, labels, perturbed, logits, restore = wrap(
            inputs, labels, perturbed, logits
        )
        a = self.a(inputs, labels, perturbed, logits)
        b = self.b(inputs, labels, perturbed, logits)
        is_adv = ep.logical_and(a, b)
        return restore(is_adv)


class Misclassification(Criterion):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __call__(self, inputs, labels, perturbed, logits):
        inputs, labels, perturbed, logits, restore = wrap(
            inputs, labels, perturbed, logits
        )
        classes = logits.argmax(axis=-1)
        is_adv = classes != labels
        return restore(is_adv)


misclassification = Misclassification()


class TargetedMisclassification(Criterion):
    def __init__(self, target_classes):
        super().__init__()
        self.target_classes = target_classes

    def __repr__(self):
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(self, inputs, labels, perturbed, logits):
        inputs, labels, perturbed, logits, restore = wrap(
            inputs, labels, perturbed, logits
        )
        classes = logits.argmax(axis=-1)
        is_adv = classes == self.target_classes
        return restore(is_adv)
