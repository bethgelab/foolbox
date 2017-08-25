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

        self.images = input_layer.input_var
        self.labels = T.ivector('labels')

        shape = lasagne.layers.get_output_shape(logits_layer)
        _, num_classes = shape
        self._num_classes = num_classes

        self.logits = lasagne.layers.get_output(logits_layer)
        self.probs = T.nnet.nnet.softmax(self.logits)

        self._batch_prediction_fn = th.function([self.images], self.logits)

        self._loss_cache = {}
        self._gradient_cache = {}
        self._loss_fn_cache = {}
        self._grad_fn_cache = {}
        self._pred_grad_fn_cache = {}

    def _loss(self, loss, **kwargs):
        import theano.tensor as T
        import lasagne
        try:
            return self._loss_cache[loss]
        except KeyError:
            if hasattr(loss, '__call__'):
                return loss(self.logits, self.labels)
            elif loss in [None, 'logits']:
                sym_loss = -self.logits[0, self.labels[0]]
            elif loss == 'crossentropy':
                sym_loss = lasagne.objectives.categorical_crossentropy(
                            self.probs, self.labels)[0]
            elif loss == 'carlini':
                sym_loss = T.max(self.logits, axis=0)[0] \
                       - self.logits[0, self.labels[0]]
                sym_loss = T.nnet.nnet.relu(sym_loss)
            else:
                raise NotImplementedError('The loss {} is currently not \
                        implemented for this framework.'.format(loss))

            self._loss_cache[loss] = sym_loss
            return sym_loss

    def _gradient(self, loss=None, **kwargs):
        import theano as th
        try:
            return self._gradient_cache[loss]
        except KeyError:
            sym_loss = self._loss(loss, **kwargs)
            print('loss shape: ', sym_loss.shape)
            sym_gradient = th.gradient.grad(sym_loss, self.images)
            print(sym_gradient.shape)

            self._gradient_cache[loss] = sym_gradient
            return sym_gradient

    def _loss_fn(self, images, labels, loss=None, **kwargs):
        import theano as th
        import theano.tensor as T
        try:
            return self._loss_fn_cache[loss](images, labels)
        except KeyError:
            sym_loss = self._loss(loss, **kwargs)
            self._loss_fn_cache[loss] = th.function([self.images, self.labels],
                                                     sym_loss)
            return self._loss_fn_cache[loss](images, labels)

    def _gradient_fn(self, images, labels, loss=None, **kwargs):
        import theano as th
        import theano.tensor as T
        try:
            return self._grad_fn_cache[loss](images, labels)
        except KeyError:
            sym_gradient = self._gradient(loss, **kwargs)
            self._grad_fn_cache[loss] = th.function([self.images, self.labels],
                                                     sym_gradient)
            return self._grad_fn_cache[loss](images, labels)

    def _predictions_and_gradient_fn(self, images, labels, loss=None, **kwargs):
        import theano as th
        import theano.tensor as T
        try:
            return self._pred_grad_fn_cache[loss](images, labels)
        except KeyError:
            sym_gradient = self._gradient(loss, **kwargs)
            self._pred_grad_fn_cache[loss] = th.function([self.images, self.labels],
                                                     [self.logits, sym_gradient])
            return self._pred_grad_fn_cache[loss](images, labels)

    def batch_predictions(self, images):
        images = self._process_input(images)
        predictions = self._batch_prediction_fn(images)
        assert predictions.shape == (images.shape[0], self.num_classes())
        return predictions

    def predictions_and_gradient(self, image, label, loss=None, **kwargs):
        image = self._process_input(image)
        label = np.array(label, dtype=np.int32)
        predictions, gradient = self._predictions_and_gradient_fn(
            image[np.newaxis], label[np.newaxis], loss, **kwargs)
        predictions = np.squeeze(predictions, axis=0)
        gradient = self._process_gradient(gradient)
        gradient = np.squeeze(gradient, axis=0)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == image.shape
        gradient = gradient.astype(image.dtype, copy=False)
        return predictions, gradient

    def gradient(self, image, label, loss=None, **kwargs):
        image = self._process_input(image)
        label = np.array(label, dtype=np.int32)
        gradient = self._gradient_fn(image[np.newaxis], label[np.newaxis],
                                     loss, **kwargs)
        gradient = self._process_gradient(gradient)
        gradient = np.squeeze(gradient, axis=0)
        assert gradient.shape == image.shape
        gradient = gradient.astype(image.dtype, copy=False)
        return gradient

    def num_classes(self):
        return self._num_classes
