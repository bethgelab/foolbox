from __future__ import absolute_import

import numpy as np
from .base import Model
from .base import DifferentiableModel


class ModelWrapper(Model):
    """Base class for models that wrap other models.

    This base class can be used to implement model wrappers
    that turn models into new models, for example by preprocessing
    the input or modifying the gradient.

    Parameters
    ----------
    model : :class:`Model`
        The model that is wrapped.

    """

    def __init__(self, model):
        super(ModelWrapper, self).__init__(
            bounds=model.bounds(),
            channel_axis=model.channel_axis())

        self.wrapped_model = model

    def __enter__(self):
        assert self.wrapped_model.__enter__() == self.wrapped_model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.wrapped_model.__exit__(exc_type, exc_value, traceback)

    def forward(self, inputs):
        return self.wrapped_model.forward(inputs)

    def num_classes(self):
        return self.wrapped_model.num_classes()


class DifferentiableModelWrapper(ModelWrapper, DifferentiableModel):
    """Base class for models that wrap other models and provide
    gradient methods.

    This base class can be used to implement model wrappers
    that turn models into new models, for example by preprocessing
    the input or modifying the gradient.

    Parameters
    ----------
    model : :class:`Model`
        The model that is wrapped.

    """

    def forward_and_gradient_one(self, x, label):
        return self.wrapped_model.forward_and_gradient_one(x, label)

    def gradient(self, inputs, labels):
        return self.wrapped_model.gradient(inputs, labels)

    def backward(self, gradient, inputs):
        return self.wrapped_model.backward(gradient, inputs)


class ModelWithoutGradients(ModelWrapper):
    """Turns a model into a model without gradients.

    """
    pass


class ModelWithEstimatedGradients(DifferentiableModelWrapper):
    """Turns a model into a model with gradients estimated
    by the given gradient estimator.

    Parameters
    ----------
    model : :class:`Model`
        The model that is wrapped.
    gradient_estimator : `callable`
        Callable taking three arguments (pred_fn, x, label) and
        returning the estimated gradients. pred_fn will be the
        forward method of the wrapped model.
    """

    def __init__(self, model, gradient_estimator):
        super(ModelWithEstimatedGradients, self).__init__(
            model=model)

        assert callable(gradient_estimator)
        self._gradient_estimator = gradient_estimator

    def forward_and_gradient_one(self, x, label):
        predictions = self.forward_one(x)
        gradient = self.gradient_one(x, label)
        return predictions, gradient

    def _gradient_one(self, x, label):
        pred_fn = self.forward
        bounds = self.bounds()
        return self._gradient_estimator(pred_fn, x, label, bounds)

    def gradient(self, inputs, labels):
        if inputs.shape[0] == labels.shape[0] == 1:
            return self._gradient_one(inputs[0], labels[0])[np.newaxis]
        raise NotImplementedError

    def backward(self, gradient, inputs):
        raise NotImplementedError


class CompositeModel(DifferentiableModel):
    """Combines predictions of a (black-box) model with the gradient of a
    (substitute) model.

    Parameters
    ----------
    forward_model : :class:`Model`
        The model that should be fooled and will be used for predictions.
    backward_model : :class:`Model`
        The model that provides the gradients.

    """

    def __init__(self, forward_model, backward_model):
        bounds = forward_model.bounds()
        assert bounds == backward_model.bounds()

        channel_axis = forward_model.channel_axis()
        assert channel_axis == backward_model.channel_axis()

        num_classes = forward_model.num_classes()
        assert num_classes == backward_model.num_classes()

        super(CompositeModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis)

        self.forward_model = forward_model
        self.backward_model = backward_model
        self._num_classes = num_classes

    def num_classes(self):
        return self._num_classes

    def forward(self, inputs):
        return self.forward_model.forward(inputs)

    def forward_and_gradient_one(self, x, label):
        predictions = self.forward_model.forward_one(x)
        gradient = self.backward_model.gradient_one(x, label)
        return predictions, gradient

    def gradient(self, inputs, labels):
        return self.backward_model.gradient(inputs, labels)

    def backward(self, gradient, inputs):
        return self.backward_model.backward(gradient, inputs)

    def __enter__(self):
        assert self.forward_model.__enter__() == self.forward_model
        assert self.backward_model.__enter__() == self.backward_model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        r1 = self.forward_model.__exit__(exc_type, exc_value, traceback)
        r2 = self.backward_model.__exit__(exc_type, exc_value, traceback)
        if r1 is None and r2 is None:
            return None
        return (r1, r2)  # pragma: no cover
