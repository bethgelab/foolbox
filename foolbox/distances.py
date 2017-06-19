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

"""
from __future__ import division
import sys
import abc
abstractmethod = abc.abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

import functools
import numpy as np
from numbers import Number


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
            bounds=None,
            value=None):

        if value is not None:
            # alternative constructor
            assert isinstance(value, Number)
            assert reference is None
            assert other is None
            assert bounds is None
            self.reference = None
            self.other = None
            self._bounds = None
            self._value = value
            self._gradient = None
        else:
            # standard constructor
            self.reference = reference
            self.other = other
            self._bounds = bounds
            self._value, self._gradient = self._calculate()

        assert self._value is not None

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @abstractmethod
    def _calculate(self):
        """Returns distance and gradient of distance w.r.t. to self.other"""
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return '{} = {:.6e}'.format(self.name(), self._value)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other.__class__ != self.__class__:
            raise TypeError('Comparisons are only possible between the same distance types.')  # noqa: E501
        return self.value == other.value

    def __lt__(self, other):
        if other.__class__ != self.__class__:
            raise TypeError('Comparisons are only possible between the same distance types.')  # noqa: E501
        return self.value < other.value


class MeanSquaredDistance(Distance):
    """Calculates the mean squared error between two images.

    """

    def _calculate(self):
        min_, max_ = self._bounds
        diff = (self.other - self.reference) / (max_ - min_)
        value = np.mean(np.square(diff))
        n = np.prod(self.reference.shape)
        gradient = 1 / n * 2 * diff / (max_ - min_)
        return value, gradient

    def __str__(self):
        return 'rel. MSE = {:.5f}  %'.format(self._value * 100)


MSE = MeanSquaredDistance


class MeanAbsoluteDistance(Distance):
    """Calculates the mean absolute error between two images.

    """

    def _calculate(self):
        min_, max_ = self._bounds
        diff = (self.other - self.reference) / (max_ - min_)
        value = np.mean(np.abs(diff))
        n = np.prod(self.reference.shape)
        gradient = 1 / n * np.sign(diff) / (max_ - min_)
        return value, gradient

    def __str__(self):
        return 'rel. MAE = {:.5f}  %'.format(self._value * 100)
