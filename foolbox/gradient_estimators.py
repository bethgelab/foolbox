import numpy as np

from .utils import batch_crossentropy


class CoordinateWiseGradientEstimator(object):
    def __init__(self, epsilon, clip=True):
        self._epsilon = epsilon
        self.clip = clip

    def _get_noise(self, shape):
        N = np.prod(shape)
        noise = np.eye(N, N, dtype=np.float32)
        noise = noise.reshape((N,) + shape)
        noise = np.concatenate([noise, -noise])
        return noise

    def __call__(self, pred_fn, x, label, bounds):
        noise = self._get_noise(x.shape)
        N = len(noise)

        min_, max_ = bounds
        scaled_epsilon = self._epsilon * (max_ - min_)

        theta = x + scaled_epsilon * noise
        if self.clip:
            theta = np.clip(theta, min_, max_)
        logits = pred_fn(theta)
        assert len(logits) == N
        loss = batch_crossentropy(label, logits)
        assert loss.shape == (N,)

        loss = loss.reshape((N,) + (1,) * x.ndim)
        gradient = np.sum(loss * noise, axis=0)
        gradient /= 2 * scaled_epsilon
        return gradient
