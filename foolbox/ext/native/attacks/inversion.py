import eagerpy as ep


class InversionAttack:
    """Creates "negative images" by inverting the pixel values according to
    [1]_.
    References
    ----------
    .. [1] Hossein Hosseini, Baicen Xiao, Mayoore Jaiswal, Radha Poovendran,
           "On the Limitation of Convolutional Neural Networks in Recognizing
           Negative Images",
            https://arxiv.org/abs/1607.02533
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, labels):
        x = ep.astensor(inputs)
        min_, max_ = self.model.bounds()
        x = min_ + max_ - x
        return x.tensor
