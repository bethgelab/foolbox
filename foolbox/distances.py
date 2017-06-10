"""
Provides classes to measure the distance between images.

Distances
---------

.. autosummary::
   :nosignatures:

   MeanSquaredDistance
   MeanAbsoluteDistance

Aliases
-------

.. autosummary::
   :nosignatures:

   MSE

Base class
----------

To implement a new distance, simply subclass the :class:`Distance` class and
implement the :meth:`_calculate` method.

.. autosummary::
   :nosignatures:

   Distance

Examples
--------

>>> import numpy as np
>>> from foolbox.distances import MeanSquaredDistance
>>> d = MeanSquaredDistance(np.array([1, 2]), np.array([2, 2]))
>>> print(d)
MSE = 5.000000e-01 (rel. MSE = 20.0 %)
>>> assert d.value() == 0.5

"""

from abc import ABC, abstractmethod
import functools
import numpy as np


@functools.total_ordering
class Distance(ABC):
    """Base class for distances.

    This class should be subclassed when implementing
    new distances. Subclasses must implement _calculate.

    """

    def __init__(
            self,
            reference=None,
            other=None,
            *,
            value=None,
            gradient=None):

        self.reference = reference
        self.other = other

        if value is not None or gradient is not None:
            assert reference is None
            assert other is None
            self._value = value
            self._gradient = gradient
        else:
            self._value, self._gradient = self._calculate()

    def value(self):
        return self._value

    def gradient(self):
        return self._gradient

    @abstractmethod
    def _calculate(self):
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return '{} = {:.6e}'.format(self.name(), self._value)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other.__class__ != self.__class__:
            raise NotImplementedError('Comparisons are only possible between the same distance types.')  # noqa: E501
        return self.value().__eq__(other.value())

    def __lt__(self, other):
        if other.__class__ != self.__class__:
            raise NotImplementedError('Comparisons are only possible between the same distance types.')  # noqa: E501
        return self.value().__lt__(other.value())


class MeanSquaredDistance(Distance):
    """Calculates the mean squared error between two images.

    """

    def _calculate(self):
        diff = self.other - self.reference
        value = np.mean(np.square(diff))
        n = np.prod(self.reference.shape)
        gradient = 1 / n * 2 * diff

        rel = value / np.mean(np.square(self.reference)) * 100
        self._rel = rel
        return value, gradient

    def __str__(self):
        try:
            return 'MSE = {:.6e} (rel. MSE = {:.1f} %)'.format(
                self._value, self._rel)
        except AttributeError:
            return 'MSE = {:.6e}'.format(
                self._value)


MSE = MeanSquaredDistance


class MeanAbsoluteDistance(Distance):
    """Calculates the mean absolute error between two images.

    """

    def _calculate(self):
        diff = self.other - self.reference
        value = np.mean(np.abs(diff))
        n = np.prod(self.reference.shape)
        gradient = 1 / n * np.sign(diff) * np.abs(diff)

        rel = value / np.mean(np.abs(self.reference)) * 100
        self._rel = rel
        return value, gradient

    def __str__(self):
        try:
            return 'MAE = {:.6e} (rel. MAE = {:.1f} %)'.format(
                self._value, self._rel)
        except AttributeError:
            return 'MAE = {:.6e}'.format(
                self._value)
