from abc import ABC
from abc import abstractmethod


class Model(ABC):
    @abstractmethod
    def bounds(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError
