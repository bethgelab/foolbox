import numpy as np

from .base import Attack
from .base import call_decorator


class PrecomputedAdversarialsAttack(Attack):
    """Attacks a model using precomputed adversarial candidates.

    Parameters
    ----------
    inputs : `numpy.ndarray`
        The original inputs that will be expected by this attack.
    outputs : `numpy.ndarray`
        The adversarial candidates corresponding to the inputs.
    *args : positional args
        Poistional args passed to the `Attack` base class.
    **kwargs : keyword args
        Keyword args passed to the `Attack` base class.
    """

    def __init__(self, inputs, outputs, *args, **kwargs):
        super(PrecomputedAdversarialsAttack, self).__init__(*args, **kwargs)

        assert inputs.shape == outputs.shape

        self._inputs = inputs
        self._outputs = outputs

    def _get_output(self, a, x):
        """ Looks up the precomputed adversarial for a given input.

        """
        sd = np.square(self._inputs - x)
        mses = np.mean(sd, axis=tuple(range(1, sd.ndim)))
        index = np.argmin(mses)

        # if we run into numerical problems with this approach, we might
        # need to add a very tiny threshold here
        if mses[index] > 0:
            raise ValueError("Could not find a precomputed adversarial for this input")
        return self._outputs[index]

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True):
        """Attacks a model using precomputed adversarial candidates.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        x = a.unperturbed
        adversarial = self._get_output(a, x)
        a.forward_one(adversarial)
