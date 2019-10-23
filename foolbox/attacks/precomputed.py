import numpy as np

from .base import Attack
from .base import generator_decorator


class PrecomputedAdversarialsAttack(Attack):
    """Attacks a model using precomputed adversarial candidates."""

    def _get_output(self, a, x, inputs, outputs):
        """ Looks up the precomputed adversarial for a given input.

        """
        sd = np.square(inputs - x)
        mses = np.mean(sd, axis=tuple(range(1, sd.ndim)))
        index = np.argmin(mses)

        # if we run into numerical problems with this approach, we might
        # need to add a very tiny threshold here
        if mses[index] > 0:
            raise ValueError(
                "Could not find a precomputed adversarial for " "this input"
            )
        return outputs[index]

    @generator_decorator
    def as_generator(self, a, candidate_inputs, candidate_outputs):
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
        candidate_inputs : `numpy.ndarray`
            The original inputs that will be expected by this attack.
        candidate_outputs : `numpy.ndarray`
            The adversarial candidates corresponding to the inputs.
        """

        assert candidate_inputs.shape == candidate_outputs.shape

        x = a.unperturbed
        adversarial = self._get_output(a, x, candidate_inputs, candidate_outputs)
        yield from a.forward_one(adversarial)
