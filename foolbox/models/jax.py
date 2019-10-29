import numpy as onp

from .base import DifferentiableModel


class JAXModel(DifferentiableModel):
    """Creates a :class:`Model` instance from a `JAX` predict function.

    Parameters
    ----------
    predict : `function`
        The JAX-compatible function that takes a batch of inputs as
        and returns a batch of predictions (logits); use
        functools.partial(predict, params) to pass params if necessary
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    num_classes : int
        Number of classes for which the model will output predictions.
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
        self, predict, bounds, num_classes, channel_axis=3, preprocessing=(0, 1)
    ):
        # lazy import
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp

        super(JAXModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocessing=preprocessing
        )

        self._num_classes = num_classes
        self._predict = predict

        def cross_entropy(logits, labels):
            assert logits.ndim == 2
            assert labels.ndim == 1
            assert len(logits) == len(labels)
            logprobs = logits - logsumexp(logits, axis=1, keepdims=True)
            nll = jnp.take_along_axis(logprobs, jnp.expand_dims(labels, axis=1), axis=1)
            ce = -jnp.mean(nll)
            return ce

        def loss(x, labels):
            logits = predict(x)
            return cross_entropy(logits, labels)

        self._loss = loss
        self._grad = jax.grad(loss)

    def num_classes(self):
        return self._num_classes

    def forward(self, inputs):
        # lazy import
        import jax.numpy as jnp

        inputs, _ = self._process_input(inputs)
        n = len(inputs)
        inputs = jnp.asarray(inputs)
        predictions = self._predict(inputs)
        predictions = onp.asarray(predictions)
        assert predictions.ndim == 2
        assert predictions.shape == (n, self.num_classes())
        return predictions

    def forward_and_gradient(self, inputs, labels):
        # lazy import
        import jax.numpy as jnp

        inputs_shape = inputs.shape
        inputs, dpdx = self._process_input(inputs)
        labels = jnp.asarray(labels)
        inputs = jnp.asarray(inputs)
        # TODO: avoid repeated forward pass for both forward and gradient
        predictions = self._predict(inputs)
        grad = self._grad(inputs, labels)

        predictions = onp.asarray(predictions)
        assert predictions.ndim == 2
        assert predictions.shape == (len(inputs), self.num_classes())

        grad = onp.asarray(grad)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == inputs_shape

        return predictions, grad

    def gradient(self, inputs, labels):
        # lazy import
        import jax.numpy as jnp

        inputs_shape = inputs.shape
        inputs, dpdx = self._process_input(inputs)
        labels = jnp.asarray(labels)
        inputs = jnp.asarray(inputs)
        grad = self._grad(inputs, labels)
        grad = onp.asarray(grad)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == inputs_shape
        return grad

    def backward(self, gradient, inputs):
        # lazy import
        import jax
        import jax.numpy as jnp

        assert gradient.ndim == 2
        gradient = jnp.asarray(gradient)

        input_shape = inputs.shape
        inputs, dpdx = self._process_input(inputs)
        inputs = jnp.asarray(inputs)

        predictions, vjp_fun = jax.vjp(self._predict, inputs)
        assert gradient.shape == predictions.shape

        (grad,) = vjp_fun(gradient)
        grad = onp.asarray(grad)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape
        return grad

    def _loss_fn(self, x, label):
        # lazy import
        import jax.numpy as jnp

        x, _ = self._process_input(x)
        inputs = jnp.asarray(x[onp.newaxis])
        labels = jnp.asarray([label])

        # if x and label were already batched, make sure that we remove
        # the added dimension again
        if len(labels.shape) == 2:
            labels = labels.squeeze(axis=0)
            inputs = inputs.squeeze(axis=0)

        loss = self._loss(inputs, labels)
        loss = onp.asarray(loss)
        return loss
