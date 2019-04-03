"""
Provides a class that represents an adversarial example.

"""

from .adversarial import Adversarial
from .adversarial import StopAttack


class YieldingAdversarial(Adversarial):
    def check_original(self):
        try:
            # TODO: this call is still done without batching
            super(YieldingAdversarial, self).predictions(
                self._Adversarial__original_image)
        except StopAttack:
            # if a threshold is specified and the original input is
            # misclassified, this can already cause a StopAttack
            # exception
            assert self.distance.value == 0.

    def predictions(self, image, strict=True, return_details=False):
        """Interface to model.predictions for attacks.

        Parameters
        ----------
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        in_bounds = self.in_bounds(image)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        predictions = yield ('predictions', image)
        is_adversarial, is_best, distance = self._Adversarial__is_adversarial(
            image, predictions, in_bounds)

        assert predictions.ndim == 1
        if return_details:
            return predictions, is_adversarial, is_best, distance
        else:
            return predictions, is_adversarial

    def batch_predictions(
            self, images, greedy=False, strict=True, return_details=False):
        raise NotImplementedError

    def gradient(self, image=None, label=None, strict=True):
        """Interface to model.gradient for attacks.

        Parameters
        ----------
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
            Defaults to the original image.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if image is None:
            image = self._Adversarial__original_image
        if label is None:
            label = self._Adversarial__original_class

        assert not strict or self.in_bounds(image)

        self._total_gradient_calls += 1
        gradient = yield ('gradient', image, label)

        assert gradient.shape == image.shape
        return gradient

    def predictions_and_gradient(
            self, image=None, label=None, strict=True, return_details=False):
        raise NotImplementedError

    def backward(self, gradient, image=None, strict=True):
        """Interface to model.backward for attacks.

        Parameters
        ----------
        gradient : `numpy.ndarray`
            Gradient of some loss w.r.t. the logits.
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).

        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient w.r.t the image.

        See Also
        --------
        :meth:`gradient`

        """
        assert self.has_gradient()
        assert gradient.ndim == 1

        if image is None:
            image = self._Adversarial__original_image

        assert not strict or self.in_bounds(image)

        self._total_gradient_calls += 1
        gradient = yield ('backward', gradient, image)

        assert gradient.shape == image.shape
        return gradient
