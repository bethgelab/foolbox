import abc
from abc import abstractmethod
import eagerpy as ep


class Criterion(abc.ABC):
    @abstractmethod
    def __call__(
        self,
        inputs: ep.Tensor,
        labels: ep.Tensor,
        perturbed: ep.Tensor,
        logits: ep.Tensor,
    ) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def __and__(self, other):
        return And(self, other)


class And(Criterion):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __repr__(self):
        return f"{self.a!r} & {self.b!r}"

    def __call__(
        self,
        inputs: ep.Tensor,
        labels: ep.Tensor,
        perturbed: ep.Tensor,
        logits: ep.Tensor,
    ) -> ep.Tensor:
        a = self.a(inputs, labels, perturbed, logits)
        b = self.b(inputs, labels, perturbed, logits)
        return ep.logical_and(a, b)


class Misclassification(Criterion):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __call__(
        self,
        inputs: ep.Tensor,
        labels: ep.Tensor,
        perturbed: ep.Tensor,
        logits: ep.Tensor,
    ) -> ep.Tensor:
        classes = logits.argmax(axis=-1)
        return classes != labels


misclassification = Misclassification()


class TargetedMisclassification(Criterion):
    def __init__(self, target_classes: ep.Tensor):
        super().__init__()
        self.target_classes = target_classes

    def __repr__(self):
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(
        self,
        inputs: ep.Tensor,
        labels: ep.Tensor,
        perturbed: ep.Tensor,
        logits: ep.Tensor,
    ) -> ep.Tensor:
        classes = logits.argmax(axis=-1)
        return classes == self.target_classes
