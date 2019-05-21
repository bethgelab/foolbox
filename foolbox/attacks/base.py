import warnings
import logging
import functools
import sys
import abc
abstractmethod = abc.abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:  # pragma: no cover
    ABC = abc.ABCMeta('ABC', (), {})

from ..adversarial import Adversarial
from ..yielding_adversarial import YieldingAdversarial
from ..adversarial import StopAttack
from ..criteria import Misclassification
from ..distances import MSE


class Attack(ABC):
    """Abstract base class for adversarial attacks.

    The :class:`Attack` class represents an adversarial attack that searches
    for adversarial examples. It should be subclassed when implementing new
    attacks.

    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
        Ignored if the attack is called with an :class:`Adversarial` instance.
    criterion : a :class:`Criterion` instance
        The criterion that determines which inputs are adversarial.
        Ignored if the attack is called with an :class:`Adversarial` instance.
    distance : a :class:`Distance` class
        The measure used to quantify similarity between inputs.
        Ignored if the attack is called with an :class:`Adversarial` instance.
    threshold : float or :class:`Distance`
        If not None, the attack will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the attack will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the attack; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the `reached_threshold()` method can be used to check
        if the threshold has been reached.
        Ignored if the attack is called with an :class:`Adversarial` instance.

    Notes
    -----
    If a subclass overwrites the constructor, it should call the super
    constructor with *args and **kwargs.

    """

    def __init__(self,
                 model=None, criterion=Misclassification(),
                 distance=MSE, threshold=None):
        self._default_model = model
        self._default_criterion = criterion
        self._default_distance = distance
        self._default_threshold = threshold

        # to customize the initialization in subclasses, please
        # try to overwrite _initialize instead of __init__ if
        # possible
        self._initialize()

    def _initialize(self):
        """Additional initializer that can be overwritten by
        subclasses without redefining the full __init__ method
        including all arguments and documentation."""
        pass

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

        if isinstance(input_or_adv, YieldingAdversarial):
            raise ValueError('If you pass an Adversarial instance, it must not be a YieldingAdversarial instance'
                             ' when calling non-batch-supporting attacks like this one (check foolbox.batch_attacks).')
        elif isinstance(input_or_adv, Adversarial):
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
                distance = self._default_distance
                threshold = self._default_threshold
                if model is None or criterion is None:
                    raise ValueError('The attack needs to be initialized'
                                     ' with a model and a criterion or it'
                                     ' needs to be called with an Adversarial'
                                     ' instance.')
                a = Adversarial(model, criterion, input_or_adv, label,
                                distance=distance, threshold=threshold)

        assert a is not None

        if a.distance.value == 0.:
            warnings.warn('Not running the attack because the original input'
                          ' is already misclassified and the adversarial thus'
                          ' has a distance of 0.')
        elif a.reached_threshold():
            warnings.warn('Not running the attack because the given treshold'
                          ' is already reached')
        else:
            try:
                _ = call_fn(self, a, label=None, unpack=None, **kwargs)
                assert _ is None, 'decorated __call__ method must return None'
            except StopAttack:
                # if a threshold is specified, StopAttack will be thrown
                # when the treshold is reached; thus we can do early
                # stopping of the attack
                logging.info('threshold reached, stopping attack')

        if a.perturbed is None:
            warnings.warn('{} did not find an adversarial, maybe the model'
                          ' or the criterion is not supported by this'
                          ' attack.'.format(self.name()))

        if unpack:
            return a.perturbed
        else:
            return a

    return wrapper
