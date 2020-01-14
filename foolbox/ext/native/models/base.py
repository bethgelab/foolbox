from abc import ABC
from abc import abstractmethod


class Model(ABC):
    @abstractmethod
    def bounds(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def gradient(self, inputs, labels):
        raise NotImplementedError

    @abstractmethod
    def value_and_grad(self, f, has_aux=False):
        raise NotImplementedError
