import numpy as np
import abc
from abc import abstractmethod


def _create_preprocessing_fn(params):
    mean, std = params
    mean = np.asarray(mean)
    std = np.asarray(std)

    def identity(x):
        return x

    if np.all(mean == 0) and np.all(std == 1):

        def preprocessing(x):
            return x, identity

    elif np.all(std == 1):

        def preprocessing(x):
            _mean = mean.astype(x.dtype)
            return x - _mean, identity

    elif np.all(mean == 0):

        def preprocessing(x):
            _std = std.astype(x.dtype)

            def grad(dmdp):
                return dmdp / _std

            return x / _std, grad

    else:

        def preprocessing(x):
            _mean = mean.astype(x.dtype)
            _std = std.astype(x.dtype)
            result = x - _mean
            result /= _std

            def grad(dmdp):
                return dmdp / _std

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
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.

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
