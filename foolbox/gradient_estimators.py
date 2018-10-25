# -*- coding: utf-8 -*-
"""
Gradient estimators to numerically approximate gradients.
"""
import logging
import warnings
import numpy as np

from .utils import batch_crossentropy
from . import nprng


class CoordinateWiseGradientEstimator(object):
    """Implements a simple gradient-estimator using
    the coordinate-wise finite-difference method.

    """
    def __init__(self, epsilon, clip=True):
        self._epsilon = epsilon
        self.clip = clip

    def _get_noise(self, shape, dtype):
        N = np.prod(shape)
        noise = np.eye(N, N, dtype=dtype)
        noise = noise.reshape((N,) + shape)
        noise = np.concatenate([noise, -noise])
        return noise

    def __call__(self, pred_fn, x, label, bounds):
        noise = self._get_noise(x.shape, x.dtype)
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
        assert loss.ndim == noise.ndim
        gradient = np.sum(loss * noise, axis=0)
        gradient /= 2 * scaled_epsilon
        return gradient


class EvolutionaryStrategiesGradientEstimator(object):
    """Implements gradient estimation using evolution strategies.

    This gradient estimator is based on work from [1]_ and [2]_.

    References
    ----------
    .. [1] Andrew Ilyas, Logan Engstrom, Anish Athalye, Jessy Lin,
           "Black-box Adversarial Attacks with Limited Queries and
           Information", https://arxiv.org/abs/1804.08598
    .. [2] Daan Wierstra, Tom Schaul, Jan Peters, Jürgen Schmidhuber,
           "Natural evolution strategies",
           http://people.idsia.ch/~tom/publications/nes.pdf

    """
    def __init__(self, epsilon, samples=100, clip=True):
        self._epsilon = epsilon
        if samples % 2 != 0:  # pragma: no cover
            warnings.warn('antithetic sampling: samples should be even')
        self._samples = (samples // 2) * 2
        self.clip = clip

    def _get_noise(self, shape, dtype):
        samples = self._samples
        assert samples % 2 == 0
        shape = (samples // 2,) + shape
        noise = nprng.normal(size=shape).astype(np.float32)
        noise = np.concatenate([noise, -noise])
        return noise

    def __call__(self, pred_fn, x, label, bounds):
        noise = self._get_noise(x.shape, x.dtype)
        N = len(noise)

        if N >= 2 * x.size:  # pragma: no cover
            logging.info('CoordinateWiseGradientEstimator might be better'
                         ' without requiring more samples.')

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
        assert loss.ndim == noise.ndim
        gradient = np.mean(loss * noise, axis=0)
        gradient /= 2 * scaled_epsilon
        return gradient
