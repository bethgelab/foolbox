import numpy as np
from .base import Model
from .base import DifferentiableModel
from ..gradient_estimators import GradientEstimatorBase


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
            bounds=model.bounds(), channel_axis=model.channel_axis()
        )

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

    def forward_and_gradient(self, x, label):
        return self.wrapped_model.forward_and_gradient(x, label)

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
    gradient_estimator : :class:`GradientEstimatorBase`
        GradientEstimator object that can estimate gradients for single and batched
        samples.
    """

    def __init__(self, model, gradient_estimator):
        super(ModelWithEstimatedGradients, self).__init__(model=model)

        assert issubclass(type(gradient_estimator), GradientEstimatorBase)
        self._gradient_estimator = gradient_estimator

    def forward_and_gradient_one(self, x, label):
        predictions = self.forward_one(x)
        gradient = self.gradient_one(x, label)
        return predictions, gradient

    def forward_and_gradient(self, inputs, labels):
        predictions = self.forward(inputs)
        gradients = self.gradient(inputs, labels)
        return predictions, gradients

    def gradient_one(self, x, label):
        pred_fn = self.forward
        bounds = self.bounds()
        return self._gradient_estimator.estimate_one(pred_fn, x, label, bounds)

    def gradient(self, inputs, labels):
        pred_fn = self.forward
        bounds = self.bounds()
        return self._gradient_estimator.estimate(pred_fn, inputs, labels, bounds)

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

        super(CompositeModel, self).__init__(bounds=bounds, channel_axis=channel_axis)

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

    def forward_and_gradient(self, inputs, labels):
        predictions = self.forward_model.forward(inputs)
        gradients = self.backward_model.gradient(inputs, labels)
        return predictions, gradients

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


class EnsembleAveragedModel(DifferentiableModelWrapper):
    """Reduces stochastic effects in networks by averaging both forward and backward
     calculations of the network by creating an ensemble of the same model and averaging
     over multiple runs (i.e. instances in the ensemble) as described in [1]_.

    References
    ----------
    .. [1] Roland S. Zimmermann,
           "Comment on 'Adv-BNN: Improved Adversarial Defense through Robust Bayesian
           Neural Network'", https://arxiv.org/abs/1907.00895

    Parameters
    ----------
    model : :class:`Model`
        The model that is wrapped.
    ensemble_size : int
        Number of networks in the ensemble over which the predictions/gradients
        will be averaged.

    """

    def __init__(self, model, ensemble_size):
        assert ensemble_size > 0, "Ensemble must contain at least 1 member."
        super(EnsembleAveragedModel, self).__init__(model=model)
        self.ensemble_size = ensemble_size

    def forward_and_gradient_one(self, x, label):
        predictions = []
        gradients = []
        for _ in range(self.ensemble_size):
            prediction, gradient = self.wrapped_model.forward_and_gradient_one(x, label)
            predictions.append(prediction)
            gradients.append(gradient)
        prediction = np.mean(predictions, axis=0)
        gradient = np.mean(gradients, axis=0)

        return prediction, gradient

    def forward_and_gradient(self, x, label):
        predictions = []
        gradients = []
        for _ in range(self.ensemble_size):
            prediction, gradient = self.wrapped_model.forward_and_gradient(x, label)
            predictions.append(prediction)
            gradients.append(gradient)
        prediction = np.mean(predictions, axis=0)
        gradient = np.mean(gradients, axis=0)

        return prediction, gradient

    def forward(self, x):
        predictions = []
        for _ in range(self.ensemble_size):
            prediction = self.wrapped_model.forward(x)
            predictions.append(prediction)
        prediction = np.mean(predictions, axis=0)

        return prediction

    def gradient(self, inputs, labels):
        gradients = []
        for _ in range(self.ensemble_size):
            gradient = self.wrapped_model.gradient(inputs, labels)
            gradients.append(gradient)
        gradient = np.mean(gradients, axis=0)

        return gradient

    def backward(self, gradient, inputs):
        grads = []
        for _ in range(self.ensemble_size):
            grad = self.wrapped_model.backward(gradient, inputs)
            grads.append(grad)
        grad = np.mean(grads, axis=0)

        return grad
