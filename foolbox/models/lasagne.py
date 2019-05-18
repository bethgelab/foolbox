from __future__ import absolute_import

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
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.

    """

    def __init__(
            self,
            input_layer,
            logits_layer,
            bounds,
            channel_axis=1,
            preprocessing=(0, 1)):

        warnings.warn('Theano is no longer being developed and Lasagne support'
                      ' in Foolbox will be removed', DeprecationWarning)

        # lazy import
        import lasagne

        inputs = input_layer.input_var
        logits = lasagne.layers.get_output(logits_layer)
        shape = lasagne.layers.get_output_shape(logits_layer)
        _, num_classes = shape

        super(LasagneModel, self).__init__(inputs, logits, bounds=bounds, num_classes=num_classes,
                                           channel_axis=channel_axis, preprocessing=preprocessing)
