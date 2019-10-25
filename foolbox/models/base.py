import numpy as np
import abc
from abc import abstractmethod
import collections


def _create_preprocessing_fn(params):
    if isinstance(params, collections.Mapping):
        mean = params.get("mean", 0)
        std = params.get("std", 1)
        axis = params.get("axis", None)
        flip_axis = params.get("flip_axis", None)
        assert (
            set(params.keys()) - {"mean", "std", "axis", "flip_axis"} == set()
        ), "unknown parameter"
    else:
        mean, std = params
        axis = None
        flip_axis = None

    mean = np.asarray(mean)
    std = np.asarray(std)

    mean = np.atleast_1d(mean)
    std = np.atleast_1d(std)

    if axis is not None:
        assert mean.ndim == 1, "If axis is specified, mean should be 1-dimensional"
        assert std.ndim == 1, "If axis is specified, std should be 1-dimensional"
        assert (
            axis < 0
        ), "axis must be negative integer, with -1 representing the last axis"
        s = (1,) * (abs(axis) - 1)
        mean = mean.reshape(mean.shape + s)
        std = std.reshape(std.shape + s)

    def identity(x):
        return x

    if flip_axis is None:
        maybe_flip = identity
    else:

        def maybe_flip(x):
            return np.flip(x, axis=flip_axis)

    if np.all(mean == 0) and np.all(std == 1):

        def preprocessing(x):
            x = maybe_flip(x)
            return x, maybe_flip

    elif np.all(std == 1):

        def preprocessing(x):
            x = maybe_flip(x)
            _mean = mean.astype(x.dtype)
            return x - _mean, maybe_flip

    elif np.all(mean == 0):

        def preprocessing(x):
            x = maybe_flip(x)
            _std = std.astype(x.dtype)

            def grad(dmdp):
                return maybe_flip(dmdp / _std)

            return x / _std, grad

    else:

        def preprocessing(x):
            x = maybe_flip(x)
            _mean = mean.astype(x.dtype)
            _std = std.astype(x.dtype)
            result = x - _mean
            result /= _std

            def grad(dmdp):
                return maybe_flip(dmdp / _std)

            return result, grad

    return preprocessing


class Model(abc.ABC):
    """Base class to provide attacks with a unified interface to models.

    The :class:`Model` class represents a model and provides a
    unified interface to its predictions. Subclasses must implement
    forward and num_classes.

    :class:`Model` instances can be used as context managers and subclasses
    can require this to allocate and release resources.

    Parameters
    ----------
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: dict or tuple
        Can be a tuple with two elements representing mean and standard
        deviation or a dict with keys "mean" and "std". The two elements
        should be floats or numpy arrays. "mean" is subtracted from the input,
        the result is then divided by "std". If "mean" and "std" are
        1-dimensional arrays, an additional (negative) "axis" key can be
        given such that "mean" and "std" will be broadcasted to that axis
        (typically -1 for "channels_last" and -3 for "channels_first", but
        might be different when using e.g. 1D convolutions). Finally,
        a (negative) "flip_axis" can be specified. This axis will be flipped
        (before "mean" is subtracted), e.g. to convert RGB to BGR.

    """

    def __init__(self, bounds, channel_axis, preprocessing=(0, 1)):
        assert len(bounds) == 2
        self._bounds = bounds
        self._channel_axis = channel_axis

        if not callable(preprocessing):
            preprocessing = _create_preprocessing_fn(preprocessing)
        assert callable(preprocessing)
        self._preprocessing = preprocessing

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def bounds(self):
        return self._bounds

    def channel_axis(self):
        return self._channel_axis

    def _process_input(self, x):
        p, grad = self._preprocessing(x)
        if hasattr(p, "dtype"):
            assert p.dtype == x.dtype
        p = np.asarray(p, dtype=x.dtype)
        assert callable(grad)
        return p, grad

    def _process_gradient(self, backward, dmdp):
        """
        backward: `callable`
            callable that backpropagates the gradient of the model w.r.t to
            preprocessed input through the preprocessing to get the gradient
            of the model's output w.r.t. the input before preprocessing
        dmdp: gradient of model w.r.t. preprocessed input
        """
        if backward is None:  # pragma: no cover
            raise ValueError(
                "Your preprocessing function does not provide"
                " an (approximate) gradient"
            )
        dmdx = backward(dmdp)
        assert dmdx.dtype == dmdp.dtype
        return dmdx

    @abstractmethod
    def forward(self, inputs):
        """Takes a batch of inputs and returns the logits predicted by the underlying model.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.

        Returns
        -------
        `numpy.ndarray`
            Predicted logits with shape (batch size, number of classes).

        See Also
        --------
        :meth:`forward_one`

        """
        raise NotImplementedError

    def forward_one(self, x):
        """Takes a single input and returns the logits predicted by the underlying model.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model (without the batch dimension).

        Returns
        -------
        `numpy.ndarray`
            Predicted logits with shape (number of classes,).

        See Also
        --------
        :meth:`forward`

        """
        return np.squeeze(self.forward(x[np.newaxis]), axis=0)

    @abstractmethod
    def num_classes(self):
        """Determines the number of classes.

        Returns
        -------
        int
            The number of classes for which the model creates predictions.

        """
        raise NotImplementedError


class DifferentiableModel(Model):
    """Base class for differentiable models.

    The :class:`DifferentiableModel` class can be used as a base class for models that can support
    gradient backpropagation. Subclasses must implement gradient and backward.

    A differentiable model does not necessarily provide reasonable values for the gradient, the gradient
    can be wrong. It only guarantees that the relevant methods can be called.

    """

    @abstractmethod
    def gradient(self, inputs, labels):
        """Takes a batch of inputs and labels and returns the gradient of the cross-entropy loss w.r.t. the inputs.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).

        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the inputs.

        See Also
        --------
        :meth:`gradient_one`
        :meth:`backward`

        """
        raise NotImplementedError

    def gradient_one(self, x, label):
        """Takes a single input and label and returns the gradient of the cross-entropy loss w.r.t. the input.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model (without the batch dimension).
        label : int
            Class label of the input as an integer in [0, number of classes).

        Returns
        -------
        `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the input.

        See Also
        --------
        :meth:`gradient`

        """
        return np.squeeze(
            self.gradient(x[np.newaxis], np.asarray(label)[np.newaxis]), axis=0
        )

    @abstractmethod
    def backward(self, gradient, inputs):
        """Backpropagates the gradient of some loss w.r.t. the logits through the underlying
        model and returns the gradient of that loss w.r.t to the inputs.

        Parameters
        ----------
        gradient : `numpy.ndarray`
            Gradient of some loss w.r.t. the logits with shape (batch size, number of classes).
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.

        Returns
        -------
        `numpy.ndarray`
            The gradient of the respective loss w.r.t the inputs.

        See Also
        --------
        :meth:`backward_one`
        :meth:`gradient`

        """
        raise NotImplementedError

    def backward_one(self, gradient, x):
        """Backpropagates the gradient of some loss w.r.t. the logits through the underlying
        model and returns the gradient of that loss w.r.t to the input.

        Parameters
        ----------
        gradient : `numpy.ndarray`
            Gradient of some loss w.r.t. the logits with shape (number of classes,).
        x : `numpy.ndarray`
            Single input with shape as expected by the model (without the batch dimension).

        Returns
        -------
        `numpy.ndarray`
            The gradient of the respective loss w.r.t the input.

        See Also
        --------
        :meth:`backward`

        """
        return np.squeeze(self.backward(gradient[np.newaxis], x[np.newaxis]), axis=0)

    @abstractmethod
    def forward_and_gradient(self, inputs, labels):
        """Takes inputs and labels and returns both the logits predicted by the underlying
        model and the gradients of the cross-entropy loss w.r.t. the inputs.


        Parameters
        ----------
        inputs : `numpy.ndarray`
            Inputs with shape as expected by the model (with the batch dimension).
        labels : `numpy.ndarray`
            Array of the class label of the inputs as an integer in [0, number of classes).

        Returns
        -------
        `numpy.ndarray`
            Predicted logits with shape (batch size, number of classes).
        `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the input.

        See Also
        --------
        :meth:`forward_one`
        :meth:`gradient_one`

        """

        raise NotImplementedError

    def forward_and_gradient_one(self, x, label):
        """Takes a single input and label and returns both the logits predicted by the underlying
        model and the gradient of the cross-entropy loss w.r.t. the input.

        Defaults to individual calls to forward_one and gradient_one but can be overriden by
        subclasses to provide a more efficient implementation.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model (without the batch dimension).
        label : int
            Class label of the input as an integer in [0, number of classes).

        Returns
        -------
        `numpy.ndarray`
            Predicted logits with shape (batch size, number of classes).
        `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the input.

        See Also
        --------
        :meth:`forward_one`
        :meth:`gradient_one`

        """
        return self.forward_one(x), self.gradient_one(x, label)  # pragma: no cover
