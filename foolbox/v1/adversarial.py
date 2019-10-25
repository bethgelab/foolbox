"""
Provides a class that represents an adversarial example.

"""
import inspect
import numpy as np
import numbers

from ..distances import Distance
from ..distances import MSE


class StopAttack(Exception):
    """Exception thrown to request early stopping of an attack
    if a given (optional!) threshold is reached."""

    pass


class Adversarial(object):
    """Defines an adversarial that should be found and stores the result.

    The :class:`Adversarial` class represents a single adversarial example
    for a given model, criterion and reference input. It can be passed to
    an adversarial attack to find the actual adversarial perturbation.

    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
    criterion : a :class:`Criterion` instance
        The criterion that determines which inputs are adversarial.
    unperturbed : a :class:`numpy.ndarray`
        The unperturbed input to which the adversarial input should be as close as possible.
    original_class : int
        The ground-truth label of the unperturbed input.
    distance : a :class:`Distance` class
        The measure used to quantify how close inputs are.
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

    """

    def __init__(
        self,
        model,
        criterion,
        unperturbed,
        original_class,
        distance=MSE,
        threshold=None,
        verbose=False,
    ):
        if inspect.isclass(criterion):
            raise ValueError("criterion should be an instance, not a class")

        self.__model = model
        self.__criterion = criterion
        self.__unperturbed = unperturbed
        self.__unperturbed_for_distance = unperturbed
        self.__original_class = original_class
        self.__distance = distance

        if threshold is not None and not isinstance(threshold, Distance):
            threshold = distance(value=threshold)
        self.__threshold = threshold

        self.verbose = verbose

        self.__best_adversarial = None
        self.__best_distance = distance(value=np.inf)
        self.__best_adversarial_output = None

        self._total_prediction_calls = 0
        self._total_gradient_calls = 0

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        # check if the original input is already adversarial
        self._check_unperturbed()

    def _check_unperturbed(self):
        try:
            self.forward_one(self.__unperturbed)
        except StopAttack:
            # if a threshold is specified and the unperturbed input is
            # misclassified, this can already cause a StopAttack
            # exception
            assert self.distance.value == 0.0

    def _reset(self):
        self.__best_adversarial = None
        self.__best_distance = self.__distance(value=np.inf)
        self.__best_adversarial_output = None

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        self._check_unperturbed()

    @property
    def perturbed(self):
        """The best adversarial example found so far."""
        return self.__best_adversarial

    @property
    def output(self):
        """The model predictions for the best adversarial found so far.

        None if no adversarial has been found.
        """
        return self.__best_adversarial_output

    @property
    def adversarial_class(self):
        """The argmax of the model predictions for the best adversarial found so far.

        None if no adversarial has been found.
        """
        if self.output is None:
            return None
        return np.argmax(self.output)

    @property
    def distance(self):
        """The distance of the adversarial input to the original input."""
        return self.__best_distance

    @property
    def unperturbed(self):
        """The original input."""
        return self.__unperturbed

    @property
    def original_class(self):
        """The class of the original input (ground-truth, not model prediction)."""
        return self.__original_class

    @property
    def _model(self):  # pragma: no cover
        """Should not be used."""
        return self.__model

    @property
    def _criterion(self):  # pragma: no cover
        """Should not be used."""
        return self.__criterion

    @property
    def _distance(self):  # pragma: no cover
        """Should not be used."""
        return self.__distance

    def set_distance_dtype(self, dtype):
        assert dtype >= self.__unperturbed.dtype
        self.__unperturbed_for_distance = self.__unperturbed.astype(dtype, copy=False)

    def reset_distance_dtype(self):
        self.__unperturbed_for_distance = self.__unperturbed

    def normalized_distance(self, x):
        """Calculates the distance of a given input x to the original input.

        Parameters
        ----------
        x : `numpy.ndarray`
            The input x that should be compared to the original input.

        Returns
        -------
        :class:`Distance`
            The distance between the given input and the original input.

        """
        return self.__distance(self.__unperturbed_for_distance, x, bounds=self.bounds())

    def reached_threshold(self):
        """Returns True if a threshold is given and the currently
        best adversarial distance is smaller than the threshold."""
        return self.__threshold is not None and self.__best_distance <= self.__threshold

    def __new_adversarial(self, x, predictions, in_bounds):
        x = x.copy()  # to prevent accidental inplace changes
        distance = self.normalized_distance(x)
        if in_bounds and self.__best_distance > distance:
            # new best adversarial
            if self.verbose:
                print("new best adversarial: {}".format(distance))

            self.__best_adversarial = x
            self.__best_distance = distance
            self.__best_adversarial_output = predictions

            self._best_prediction_calls = self._total_prediction_calls
            self._best_gradient_calls = self._total_gradient_calls

            if self.reached_threshold():
                raise StopAttack

            return True, distance
        return False, distance

    def __is_adversarial(self, x, predictions, in_bounds):
        """Interface to criterion.is_adverarial that calls
        __new_adversarial if necessary.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            The input that should be checked.
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some input x.
        label : int
            The label of the unperturbed reference input.

        """
        is_adversarial = self.__criterion.is_adversarial(
            predictions, self.__original_class
        )
        assert isinstance(is_adversarial, bool) or isinstance(is_adversarial, np.bool_)
        if is_adversarial:
            is_best, distance = self.__new_adversarial(x, predictions, in_bounds)
        else:
            is_best = False
            distance = None
        return is_adversarial, is_best, distance

    @property
    def target_class(self):
        """Interface to criterion.target_class for attacks.

        """
        try:
            target_class = self.__criterion.target_class()
        except AttributeError:
            target_class = None
        return target_class

    def num_classes(self):
        n = self.__model.num_classes()
        assert isinstance(n, numbers.Number)
        return n

    def bounds(self):
        min_, max_ = self.__model.bounds()
        assert isinstance(min_, numbers.Number)
        assert isinstance(max_, numbers.Number)
        assert min_ < max_
        return min_, max_

    def in_bounds(self, input_):
        min_, max_ = self.bounds()
        return min_ <= input_.min() and input_.max() <= max_

    def channel_axis(self, batch):
        """Interface to model.channel_axis for attacks.

        Parameters
        ----------
        batch : bool
            Controls whether the index of the axis for a batch of inputs
            (4 dimensions) or a single input (3 dimensions) should be returned.

        """
        axis = self.__model.channel_axis()
        if not batch:
            axis = axis - 1
        return axis

    def has_gradient(self):
        """Returns true if _backward and _forward_backward can be called
        by an attack, False otherwise.

        """
        try:
            self.__model.gradient
            self.__model.gradient_one
            self.__model.forward_and_gradient_one
            self.__model.backward
            self.__model.backward_one
        except AttributeError:
            return False
        else:
            return True

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
        predictions = self.__model.forward_one(x)
        is_adversarial, is_best, distance = self.__is_adversarial(
            x, predictions, in_bounds
        )

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
        predictions = self.__model.forward(inputs)

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
            is_adversarial, is_best, distance = self.__is_adversarial(
                inputs[i], predictions[i], in_bounds_i
            )
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
            x = self.__unperturbed
        if label is None:
            label = self.__original_class

        assert not strict or self.in_bounds(x)

        self._total_gradient_calls += 1
        gradient = self.__model.gradient_one(x, label)

        assert gradient.shape == x.shape
        return gradient

    def forward_and_gradient_one(
        self, x=None, label=None, strict=True, return_details=False
    ):
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
            x = self.__unperturbed
        if label is None:
            label = self.__original_class

        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        self._total_gradient_calls += 1
        predictions, gradient = self.__model.forward_and_gradient_one(x, label)
        is_adversarial, is_best, distance = self.__is_adversarial(
            x, predictions, in_bounds
        )

        assert predictions.ndim == 1
        assert gradient.shape == x.shape
        if return_details:
            return predictions, gradient, is_adversarial, is_best, distance
        else:
            return predictions, gradient, is_adversarial

    def forward_and_gradient(self, x, label=None, strict=True, return_details=False):
        """Interface to model.forward_and_gradient_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Multiple input with shape as expected by the model
            (with the batch dimension).
        label : `numpy.ndarray`
            Labels used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if label is None:
            label = np.ones(len(x), dtype=np.int) * self.__original_class

        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += len(x)
        self._total_gradient_calls += len(x)
        predictions, gradients = self.__model.forward_and_gradient(x, label)

        assert predictions.ndim == 2
        assert gradients.shape == x.shape

        is_adversarials, is_bests, distances = [], [], []
        for single_x, prediction in zip(x, predictions):
            is_adversarial, is_best, distance = self.__is_adversarial(
                single_x, prediction, in_bounds
            )
            is_adversarials.append(is_adversarial)
            is_bests.append(is_best)
            distances.append(distance)

        is_adversarials = np.array(is_adversarials)
        is_bests = np.array(is_bests)
        distances = np.array(distances)

        if return_details:
            return predictions, gradients, is_adversarials, is_bests, distances
        else:
            return predictions, gradients, is_adversarials

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
            x = self.__unperturbed

        assert not strict or self.in_bounds(x)

        self._total_gradient_calls += 1
        gradient = self.__model.backward_one(gradient, x)

        assert gradient.shape == x.shape
        return gradient
