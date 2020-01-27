from abc import ABC, abstractmethod


class Attack(ABC):
    @abstractmethod
    def __call__(self, inputs, labels):
        raise NotImplementedError  # pragma: no cover
