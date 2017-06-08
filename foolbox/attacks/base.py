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
            find=None,
            *,
            image=None,
            label=None,
            unpack=True,
            **kwargs):

        """Applies the attack.

        Parameters
        ----------
        find : :class:`Adversarial`
            The definition of the adversarial that should be found.
            If find is passed, image and label must be None.
        image : `numpy.ndarray`
            The original, correctly classified image. If image is passed,
            label must be passed as well and find must be None.
        label : int
            The reference label of the original image. If label is passed,
            image must be passed as well and find must be None.
        kwargs : dict
            Addtional keyword arguments passed to the attack.

        Notes
        -----
        Subclasses should not overwrite this method, but instead implement the
        attack in the :meth:_apply method.

        """

        if find is None:
            if image is None or label is None:
                raise ValueError('Either find or both image and label must be passed.')  # noqa: E501
            else:
                model = self.__default_model
                criterion = self.__default_criterion
                if model is None or criterion is None:
                    raise ValueError('Passing image and label is only supported if the attack was initialized with a default model and a default criterion.')  # noqa: E501

                find = Adversarial(model, criterion, image, label)
        else:
            if image is not None or label is not None:
                raise ValueError('If find is passed, image and label must be None')  # noqa: E501

        assert find is not None

        adversarial = find
        _ = self._apply(adversarial, **kwargs)
        assert _ is None, '_apply must return None'

        if unpack:
            return adversarial.get()
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
