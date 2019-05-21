"""
Provides a class that represents an adversarial example.

"""

import numpy as np

from .adversarial import Adversarial
from .adversarial import StopAttack


class YieldingAdversarial(Adversarial):
    def _check_unperturbed(self):
        try:
            # for now, we use the non-yielding implementation in the super-class
            # TODO: add support for batching this first call as well
            super(YieldingAdversarial, self).forward_one(self._Adversarial__unperturbed)
        except StopAttack:
            # if a threshold is specified and the unperturbed input is
            # misclassified, this can already cause a StopAttack
            # exception
            assert self.distance.value == 0.

    def forward_one(self, x, strict=True, return_details=False):
        """Interface to model.forward_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        predictions = yield ('forward_one', x)
        is_adversarial, is_best, distance = self._Adversarial__is_adversarial(
            x, predictions, in_bounds)

        assert predictions.ndim == 1
        if return_details:
            return predictions, is_adversarial, is_best, distance
        else:
            return predictions, is_adversarial

    def forward(self, inputs, greedy=False, strict=True, return_details=False):
        """Interface to model.forward for attacks.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the model.
        greedy : bool
            Whether the first adversarial should be returned.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        if strict:
            in_bounds = self.in_bounds(inputs)
            assert in_bounds

        self._total_prediction_calls += len(inputs)
        predictions = yield ('forward', inputs)

        assert predictions.ndim == 2
        assert predictions.shape[0] == inputs.shape[0]

        if return_details:
            assert greedy

        adversarials = []
        for i in range(len(predictions)):
            if strict:
                in_bounds_i = True
            else:
                in_bounds_i = self.in_bounds(inputs[i])
            is_adversarial, is_best, distance = self._Adversarial__is_adversarial(
                inputs[i], predictions[i], in_bounds_i)
            if is_adversarial and greedy:
                if return_details:
                    return predictions, is_adversarial, i, is_best, distance
                else:
                    return predictions, is_adversarial, i
            adversarials.append(is_adversarial)

        if greedy:  # pragma: no cover
            # no adversarial found
            if return_details:
                return predictions, False, None, False, None
            else:
                return predictions, False, None

        is_adversarial = np.array(adversarials)
        assert is_adversarial.ndim == 1
        assert is_adversarial.shape[0] == inputs.shape[0]
        return predictions, is_adversarial

    def gradient_one(self, x=None, label=None, strict=True):
        """Interface to model.gradient_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
            Defaults to the original input.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if x is None:
            x = self._Adversarial__unperturbed
        if label is None:
            label = self._Adversarial__original_class

        assert not strict or self.in_bounds(x)

        self._total_gradient_calls += 1
        gradient = yield ('gradient_one', x, label)

        assert gradient.shape == x.shape
        return gradient

    def forward_and_gradient_one(self, x=None, label=None, strict=True, return_details=False):
        """Interface to model.forward_and_gradient_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
            Defaults to the original input.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if x is None:
            x = self._Adversarial__unperturbed
        if label is None:
            label = self._Adversarial__original_class

        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        self._total_gradient_calls += 1
        predictions, gradient = yield ('forward_and_gradient_one', x, label)
        is_adversarial, is_best, distance = self._Adversarial__is_adversarial(x, predictions, in_bounds)

        assert predictions.ndim == 1
        assert gradient.shape == x.shape
        if return_details:
            return predictions, gradient, is_adversarial, is_best, distance
        else:
            return predictions, gradient, is_adversarial

    def backward_one(self, gradient, x=None, strict=True):
        """Interface to model.backward_one for attacks.

        Parameters
        ----------
        gradient : `numpy.ndarray`
            Gradient of some loss w.r.t. the logits.
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).

        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient w.r.t the input.

        See Also
        --------
        :meth:`gradient`

        """
        assert self.has_gradient()
        assert gradient.ndim == 1

        if x is None:
            x = self._Adversarial__unperturbed

        assert not strict or self.in_bounds(x)

        self._total_gradient_calls += 1
        gradient = yield ('backward_one', gradient, x)

        assert gradient.shape == x.shape
        return gradient
