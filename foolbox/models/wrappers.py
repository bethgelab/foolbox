from .base import Model


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
        super().__init__(
            bounds=model.bounds(),
            channel_axis=model.channel_axis())

        self.wrapped_model = model

    def __enter__(self):
        assert self.wrapped_model.__enter__() == self.wrapped_model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.wrapped_model.__exit__(exc_type, exc_value, traceback)


class GradientLess(ModelWrapper):
    """
    Turns a model into a model without gradients.
    """

    def batch_predictions(self, images):
        return self.wrapped_model.batch_predictions(images)

    def num_classes(self):
        return self.wrapped_model.num_classes()
