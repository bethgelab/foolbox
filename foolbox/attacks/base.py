import warnings
import functools
import sys
import abc
abstractmethod = abc.abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:  # pragma: no cover
    ABC = abc.ABCMeta('ABC', (), {})

from ..adversarial import Adversarial
from ..criteria import Misclassification


class Attack(ABC):
    """Abstract base class for adversarial attacks.

    The :class:`Attack` class represents an adversarial attack that searches
    for adversarial examples. It should be subclassed when implementing new
    attacks.

    Parameters
    ----------
    model : :class:`adversarial.Model`
        The default model to which the attack is applied if it is not called
        with an :class:`Adversarial` instance.
    criterion : :class:`adversarial.Criterion`
        The default criterion that defines what is adversarial if the attack
        is not called with an :class:`Adversarial` instance.

    Notes
    -----
    If a subclass overwrites the constructor, it should call the super
    constructor with *args and **kwargs.

    """

    def __init__(self, model=None, criterion=Misclassification()):
        self._default_model = model
        self._default_criterion = criterion

    @abstractmethod
    def __call__(self, input_or_adv, label=None, unpack=True, **kwargs):
        raise NotImplementedError

    def name(self):
        """Returns a human readable name that uniquely identifies
        the attack with its hyperparameters.

        Returns
        -------
        str
            Human readable name that uniquely identifies the attack
            with its hyperparameters.

        Notes
        -----
        Defaults to the class name but subclasses can provide more
        descriptive names and must take hyperparameters into account.

        """
        return self.__class__.__name__


def call_decorator(call_fn):
    @functools.wraps(call_fn)
    def wrapper(self, input_or_adv, label=None, unpack=True, **kwargs):
        assert input_or_adv is not None

        if isinstance(input_or_adv, Adversarial):
            a = input_or_adv
            if label is not None:
                raise ValueError('Label must not be passed when input_or_adv'
                                 ' is an Adversarial instance')
        else:
            if label is None:
                raise ValueError('Label must be passed when input_or_adv is'
                                 ' not an Adversarial instance')
            else:
                model = self._default_model
                criterion = self._default_criterion
                if model is None or criterion is None:
                    raise ValueError('The attack needs to be initialized'
                                     ' with a model and a criterion or it'
                                     ' needs to be called with an Adversarial'
                                     ' instance.')
                a = Adversarial(model, criterion, input_or_adv, label)

        assert a is not None

        if a.distance.value == 0.:
            warnings.warn('Not running the attack because the original input'
                          ' is already misclassified and the adversarial thus'
                          ' has a distance of 0.')
        else:
            _ = call_fn(self, a, label=None, unpack=None, **kwargs)
            assert _ is None, 'decorated __call__ method must return None'

        if a.image is None:
            warnings.warn('{} did not find an adversarial, maybe the model'
                          ' or the criterion is not supported by this'
                          ' attack.'.format(self.name()))

        if unpack:
            return a.image
        else:
            return a

    return wrapper
