import logging
from abc import ABC, abstractmethod

from ..adversarial import Adversarial
from ..criteria import Misclassification


class Attack(ABC):
    """Abstract base class for adversarial attacks.

    The :class:`Attack` class represents an adversarial attack that searches
    for adversarial examples. It should be subclassed when implementing new
    attacks. Subclasses must implement the _apply method.

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
        self.__default_model = model
        self.__default_criterion = criterion

    def __call__(
            self,
            image,
            label=None,
            *,
            unpack=True,
            **kwargs):

        """Applies the attack.

        Parameters
        ----------
        image : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified image. If image is a
            numpy array, label must be passed as well. If image is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original image. Must be passed
            if image is a numpy array, must not be passed if image is
            an :class:`Adversarial` instance.
        kwargs : dict
            Addtional keyword arguments passed to the attack.

        Notes
        -----
        Subclasses should not overwrite this method, but instead implement the
        attack in the :meth:_apply method.

        """

        assert image is not None

        if isinstance(image, Adversarial):
            if label is not None:
                raise ValueError('Label must not be passed when image is an Adversarial instance')  # noqa: E501k
            else:
                find = image
        else:
            if label is None:
                raise ValueError('Label must be passed when image is not an Adversarial instance')  # noqa: E501k
            else:
                model = self.__default_model
                criterion = self.__default_criterion
                if model is None or criterion is None:
                    raise ValueError('The attack needs to be initialized with a model and a criterion or it needs to be called with an Adversarial instance.')  # noqa: E501
                find = Adversarial(model, criterion, image, label)

        assert find is not None

        adversarial = find

        if adversarial.distance.value() == 0.:
            logging.info('Not running the attack because the original image is already misclassified and the adversarial thus has a distance of 0.')  # noqa: E501
        else:
            _ = self._apply(adversarial, **kwargs)
            assert _ is None, '_apply must return None'

        if adversarial.image is None:
            logging.warn('{} did not find an adversarial, maybe the model or the criterion is not supported by this attack.'.format(self.name()))  # noqa: E501

        if unpack:
            return adversarial.image
        else:
            return adversarial

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

    @abstractmethod
    def _apply(self, a):
        """Searches an adversarial example.

        To implement an attack, subclasses should implement this
        method.

        Parameters
        ----------
        a : :class:`Adversarial`
            The object that provides access to the model, criterion, original
            image, best adversarial so far, etc.

        """
        raise NotImplementedError
