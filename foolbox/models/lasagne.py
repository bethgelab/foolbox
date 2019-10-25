import warnings

from .theano import TheanoModel


class LasagneModel(TheanoModel):
    """Creates a :class:`Model` instance from a `Lasagne` network.

    Parameters
    ----------
    input_layer : `lasagne.layers.Layer`
        The input to the model.
    logits_layer : `lasagne.layers.Layer`
        The output of the model, before the softmax.
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

    def __init__(
        self, input_layer, logits_layer, bounds, channel_axis=1, preprocessing=(0, 1)
    ):

        warnings.warn(
            "Theano is no longer being developed and Lasagne support"
            " in Foolbox will be removed",
            DeprecationWarning,
        )

        # lazy import
        import lasagne

        inputs = input_layer.input_var
        logits = lasagne.layers.get_output(logits_layer)
        shape = lasagne.layers.get_output_shape(logits_layer)
        _, num_classes = shape

        super(LasagneModel, self).__init__(
            inputs,
            logits,
            bounds=bounds,
            num_classes=num_classes,
            channel_axis=channel_axis,
            preprocessing=preprocessing,
        )
