"""
Provides classes that define what is adversarial.

Criteria
--------

We provide criteria for untargeted and targeted adversarial attacks.

.. autosummary::
   :nosignatures:

   Misclassification
   TopKMisclassification
   OriginalClassProbability

.. autosummary::
   :nosignatures:

   TargetClass
   TargetClassProbability

Examples
--------

Untargeted criteria:

>>> from foolbox.criteria import Misclassification
>>> criterion1 = Misclassification()

>>> from foolbox.criteria import TopKMisclassification
>>> criterion2 = TopKMisclassification(k=5)

Targeted criteria:

>>> from foolbox.criteria import TargetClass
>>> criterion3 = TargetClass(22)

>>> from foolbox.criteria import TargetClassProbability
>>> criterion4 = TargetClassProbability(22, p=0.99)

Criteria can be combined to create a new criterion:

>>> criterion5 = criterion2 & criterion3

"""

from abc import ABC, abstractmethod

import numpy as np

from .utils import softmax


class Criterion(ABC):
    """Base class for criteria that define what is adversarial.

    The :class:`Criterion` class represents a criterion used to
    determine if predictions for an image are adversarial given
    a reference label. It should be subclassed when implementing
    new criteria. Subclasses must implement is_adversarial.

    """

    def name(self):
        """Returns a human readable name that uniquely identifies
        the criterion with its hyperparameters.

        Returns
        -------
        str
            Human readable name that uniquely identifies the criterion
            with its hyperparameters.

        Notes
        -----
        Defaults to the class name but subclasses can provide more
        descriptive names and must take hyperparameters into account.

        """
        return self.__class__.__name__

    @abstractmethod
    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial given
        a reference label.

        Parameters
        ----------
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some image.
        label : int
            The label of the unperturbed reference image.

        Returns
        -------
        bool
            True if an image with the given predictions is an adversarial
            example when the ground-truth class is given by label, False
            otherwise.

        """
        raise NotImplementedError

    def __and__(self, other):
        return CombinedCriteria(self, other)


class CombinedCriteria(Criterion):
    """Meta criterion that combines several criteria into a new one.

    Considers images as adversarial that are considered adversarial
    by all sub-criteria that are combined by this criterion.

    Instead of using this class directly, it is possible to combine
    criteria like this: criteria1 & criteria2

    Parameters
    ----------
    *criteria : variable length list of :class:`Criterion` instances
        List of sub-criteria that will be combined.

    Notes
    -----
    This class uses lazy evaluation of the criteria in the order they
    are passed to the constructor.

    """

    def __init__(self, *criteria):
        super().__init__()
        self._criteria = criteria

    def name(self):
        """Concatenates the names of the given criteria in alphabetical order.

        If a sub-criterion is itself a combined criterion, its name is
        first split into the individual names and the names of the
        sub-sub criteria is used instead of the name of the sub-criterion.
        This is done recursively to ensure that the order and the hierarchy
        of the criteria does not influence the name.

        Returns
        -------
        str
            The alphabetically sorted names of the sub-criteria concatenated
            using double underscores between them.

        """
        names = (criterion.name() for criterion in self._criteria)
        return '__'.join(sorted(names))

    def is_adversarial(self, predictions, label):
        for criterion in self._criteria:
            if not criterion.is_adversarial(predictions, label):
                # lazy evaluation
                return False
        return True


class Misclassification(Criterion):
    """Defines adversarials as images for which the predicted class
    is not the original class.

    See Also
    --------
    :class:`TopKMisclassification`

    Notes
    -----
    Uses `numpy.argmax` to break ties.

    """

    def name(self):
        return 'Top1Misclassification'

    def is_adversarial(self, predictions, label):
        top1 = np.argmax(predictions)
        return top1 != label


class TopKMisclassification(Criterion):
    """Defines adversarials as images for which the original class is
    not one of the top k predicted classes.

    For k = 1, the :class:`Misclassification` class provides a more
    efficient implementation.

    Parameters
    ----------
    k : int
        Number of top predictions to which the reference label is
        compared to.

    See Also
    --------
    :class:`Misclassification` : Provides a more effcient implementation
        for k = 1.

    Notes
    -----
    Uses `numpy.argsort` to break ties.

    """

    def __init__(self, *, k):
        super().__init__()
        self.k = k

    def name(self):
        return 'Top{}Misclassification'.format(self.k)

    def is_adversarial(self, predictions, label):
        topk = np.argsort(predictions)[-self.k:]
        return label not in topk


class TargetClass(Criterion):
    """Defines adversarials as images for which the predicted class
    is the given target class.

    Parameters
    ----------
    target_class : int
        The target class that needs to be predicted for an image
        to be considered an adversarial.

    Notes
    -----
    Uses `numpy.argmax` to break ties.

    """

    def __init__(self, target_class):
        super().__init__()
        self._target_class = target_class

    def target_class(self):
        return self._target_class

    def name(self):
        return '{}-{}'.format(self.__class__.__name__, self.target_class())

    def is_adversarial(self, predictions, label):
        top1 = np.argmax(predictions)
        return top1 == self.target_class()


class OriginalClassProbability(Criterion):
    """Defines adversarials as images for which the probability
    of the original class is below a given threshold.

    This criterion alone does not guarantee that the class
    predicted for the adversarial image is not the original class
    (unless p < 1 / number of classes). Therefore, it should usually
    be combined with a classifcation criterion.

    Parameters
    ----------
    p : float
        The threshold probability. If the probability of the
        original class is below this threshold, the image is
        considered an adversarial. It must satisfy 0 <= p <= 1.

    """

    def __init__(self, p):
        super().__init__()
        assert 0 <= p <= 1
        self.p = p

    def name(self):
        return '{}-{:.04f}'.format(self.__class__.__name__, self.p)

    def is_adversarial(self, predictions, label):
        probabilities = softmax(predictions)
        return probabilities[label] < self.p


class TargetClassProbability(Criterion):
    """Defines adversarials as images for which the probability
    of a given target class is above a given threshold.

    If the threshold is below 0.5, this criterion does not guarantee
    that the class predicted for the adversarial image is not the
    original class. In that case, it should usually be combined with
    a classification criterion.

    Parameters
    ----------
    target_class : int
        The target class for which the predicted probability must
        be above the threshold probability p, otherwise the image
        is not considered an adversarial.
    p : float
        The threshold probability. If the probability of the
        target class is above this threshold, the image is
        considered an adversarial. It must satisfy 0 <= p <= 1.

    """

    def __init__(self, target_class, *, p):
        super().__init__()
        self._target_class = target_class
        assert 0 <= p <= 1
        self.p = p

    def target_class(self):
        return self._target_class

    def name(self):
        return '{}-{}-{:.04f}'.format(
            self.__class__.__name__, self.target_class(), self.p)

    def is_adversarial(self, predictions, label):
        probabilities = softmax(predictions)
        return probabilities[self.target_class()] > self.p
