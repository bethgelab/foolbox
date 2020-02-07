from typing import TypeVar, Any
from abc import ABC, abstractmethod
import eagerpy as ep


T = TypeVar("T")


class Criterion(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def __call__(self, perturbed: T, outputs: T) -> T:
        ...

    def __and__(self, other: "Criterion") -> "_And":
        return _And(self, other)


class _And(Criterion):
    def __init__(self, a: Criterion, b: Criterion) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def __repr__(self) -> str:
        return f"{self.a!r} & {self.b!r}"

    def __call__(self, perturbed: T, outputs: T) -> T:
        args, restore_type = ep.astensors_(perturbed, outputs)
        a = self.a(*args)
        b = self.b(*args)
        is_adv = ep.logical_and(a, b)
        return restore_type(is_adv)


class Misclassification(Criterion):
    def __init__(self, labels: Any) -> None:
        super().__init__()
        self.labels: ep.Tensor = ep.astensor(labels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs

        classes = outputs_.argmax(axis=-1)
        is_adv = classes != self.labels
        return restore_type(is_adv)


class TargetedMisclassification(Criterion):
    def __init__(self, target_classes: Any) -> None:
        super().__init__()
        self.target_classes: ep.Tensor = ep.astensor(target_classes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs

        classes = outputs_.argmax(axis=-1)
        is_adv = classes == self.target_classes
        return restore_type(is_adv)
