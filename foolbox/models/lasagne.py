from __future__ import absolute_import
import numpy as np


from .base import DifferentiableModel


class LasagneModel(DifferentiableModel):
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

        super(LasagneModel, self).__init__(bounds=bounds,
                                           channel_axis=channel_axis,
                                           preprocessing=preprocessing)

        # delay import until class is instantiated
        import theano as th
        import theano.tensor as T
        import lasagne

        images = input_layer.input_var
        labels = T.ivector('labels')
        bw_gradient_pre = T.fmatrix('bw_gradient_pre')

        shape = lasagne.layers.get_output_shape(logits_layer)
        _, num_classes = shape
        self._num_classes = num_classes

        logits = lasagne.layers.get_output(logits_layer)

        probs = T.nnet.nnet.softmax(logits)

        loss = lasagne.objectives.categorical_crossentropy(
            probs, labels)
        gradient = th.gradient.grad(loss[0], images)

        bw_loss = (logits * bw_gradient_pre).sum()
        bw_gradient = th.gradient.grad(bw_loss, images)

        self._batch_prediction_fn = th.function([images], logits)
        self._predictions_and_gradient_fn = th.function(
            [images, labels], [logits, gradient])
        self._gradient_fn = th.function([images, labels], gradient)
        self._loss_fn = th.function([images, labels], loss)
        self._bw_gradient_fn = th.function(
            [bw_gradient_pre, images], bw_gradient)

    def batch_predictions(self, images):
        predictions = self._batch_prediction_fn(images)
        assert predictions.shape == (images.shape[0], self.num_classes())
        return predictions

    def predictions_and_gradient(self, image, label):
        label = np.array(label, dtype=np.int32)
        predictions, gradient = self._predictions_and_gradient_fn(
            image[np.newaxis], label[np.newaxis])
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == image.shape
        gradient = gradient.astype(image.dtype, copy=False)
        return predictions, gradient

    def gradient(self, image, label):
        label = np.array(label, dtype=np.int32)
        gradient = self._gradient_fn(image[np.newaxis], label[np.newaxis])
        gradient = np.squeeze(gradient, axis=0)
        assert gradient.shape == image.shape
        gradient = gradient.astype(image.dtype, copy=False)
        return gradient

    def num_classes(self):
        return self._num_classes

    def backward(self, gradient, image):
        assert gradient.ndim == 1
        gradient = self._bw_gradient_fn(
            gradient[np.newaxis], image[np.newaxis])
        gradient = np.squeeze(gradient, axis=0)
        assert gradient.shape == image.shape
        gradient = gradient.astype(image.dtype, copy=False)
        assert gradient.shape == image.shape
        return gradient
