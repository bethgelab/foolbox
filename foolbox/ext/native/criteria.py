from abc import ABC, abstractmethod
from typing import TypeVar
import eagerpy as ep


T = TypeVar("T")


class Criterion(ABC):
    @abstractmethod
    def __repr__(self):
        ...

    @abstractmethod
    def __call__(self, inputs: T, labels: T, perturbed: T, logits: T) -> T:
        ...

    def __and__(self, other):
        return _And(self, other)


class _And(Criterion):
    def __init__(self, a: Criterion, b: Criterion):
        super().__init__()
        self.a = a
        self.b = b

    def __repr__(self):
        return f"{self.a!r} & {self.b!r}"

    def __call__(self, inputs: T, labels: T, perturbed: T, logits: T) -> T:
        args, restore_type = ep.astensors_(inputs, labels, perturbed, logits)
        a = self.a(*args)
        b = self.b(*args)
        is_adv = ep.logical_and(a, b)
        return restore_type(is_adv)


class Misclassification(Criterion):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __call__(self, inputs: T, labels: T, perturbed: T, logits: T) -> T:
        (labels_, logits_), restore_type = ep.astensors_(labels, logits)
        del inputs, labels, perturbed, logits

        classes = logits_.argmax(axis=-1)
        is_adv = classes != labels_
        return restore_type(is_adv)


misclassification = Misclassification()


class TargetedMisclassification(Criterion):
    def __init__(self, target_classes) -> None:
        super().__init__()
        self.target_classes: ep.Tensor = ep.astensor(target_classes)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(self, inputs: T, labels: T, perturbed: T, logits: T) -> T:
        logits_, restore_type = ep.astensor_(logits)

        classes = logits_.argmax(axis=-1)
        is_adv = classes == self.target_classes
        return restore_type(is_adv)
