import numpy as np
import eagerpy as ep

from scipy.ndimage.filters import gaussian_filter

from ..utils import atleast_kd


class GaussianBlurAttack:
    def __init__(self, model, channel_axis):
        self.model = model
        self.channel_axis = channel_axis

    def __call__(self, inputs, labels, *, steps=1000):
        x = ep.astensor(inputs)
        y = ep.astensor(labels)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 1

        assert x.ndim == 4
        if self.channel_axis == 1:
            h, w = x.shape[2:4]
        elif self.channel_axis == 3:
            h, w = x.shape[1:3]
        else:
            raise ValueError("expected 'channel_axis' to be 1 or 3, got {channel_axis}")

        size = max(h, w)

        min_, max_ = self.model.bounds()

        x0 = x
        x0np = x0.numpy()

        epsilons = np.linspace(0, 1, num=steps + 1)[1:]

        logits = ep.astensor(self.model.forward(x0.tensor))
        classes = logits.argmax(axis=-1)
        is_adv = classes != labels
        found = is_adv

        result = x0

        for epsilon in epsilons:
            # TODO: reduce the batch size to the ones that haven't been sucessful

            sigmas = [epsilon * size] * 4
            sigmas[0] = 0
            sigmas[self.channel_axis] = 0

            # TODO: once we can implement gaussian_filter in eagerpy, avoid converting from numpy
            x = gaussian_filter(x0np, sigmas)
            x = np.clip(x, min_, max_)
            x = ep.from_numpy(x0, x)

            logits = ep.astensor(self.model.forward(x.tensor))
            classes = logits.argmax(axis=-1)
            is_adv = classes != labels

            new_adv = ep.logical_and(is_adv, found.logical_not())
            result = ep.where(atleast_kd(new_adv, x.ndim), x, result)
            found = ep.logical_or(new_adv, found)

            if found.all():
                break

        return result.tensor
