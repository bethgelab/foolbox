from .base import Attack
from .base import generator_decorator


class InversionAttack(Attack):
    """Creates "negative images" by inverting the pixel values according to
    [1]_.

    References
    ----------
    .. [1] Hossein Hosseini, Baicen Xiao, Mayoore Jaiswal, Radha Poovendran,
           "On the Limitation of Convolutional Neural Networks in Recognizing
           Negative Images",
            https://arxiv.org/abs/1607.02533

    """

    @generator_decorator
    def as_generator(self, a):
        """Creates "negative images" by inverting the pixel values.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        """

        assert a.target_class is None, "This is an untargeted attack by design."

        min_, max_ = a.bounds()
        x = min_ + max_ - a.unperturbed
        yield from a.forward_one(x)
