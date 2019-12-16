import numpy as np
import eagerpy as ep

from ..utils import atleast_kd

import warnings


class LinearSearchBlendedUniformNoiseAttack:
    """Blends the input with a uniform noise input until it is misclassified."""

    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, labels, *, directions=1000, steps=1000):
        x = ep.astensor(inputs)
        min_, max_ = self.model.bounds()

        N = len(x)

        assert directions >= 1
        for j in range(directions):
            # random noise inputs tend to be classified into the same class,
            # so we might need to make very many draws if the original class
            # is that one
            random_ = ep.uniform(x, x.shape, min_, max_)
            logits_ = ep.astensor(self.model.forward(random_.tensor))
            classes_ = logits_.argmax(axis=-1)
            is_adv_ = atleast_kd(classes_ != labels, x.ndim)

            if j == 0:
                random = random_
                is_adv = is_adv_
            else:
                cond1 = is_adv.astype(x.dtype)
                cond2 = is_adv_.astype(x.dtype)
                random = cond1 * random + (1 - cond1) * cond2 * random_
                is_adv = is_adv.logical_or(is_adv_)

            if is_adv.all():
                break

        if not is_adv.all():
            warnings.warn(
                f"{self.__class__.__name__} failed to draw sufficent random"
                " inputs that are adversarial ({is_adv.sum()} / {N})."
            )

        x0 = x

        npdtype = x.numpy().dtype
        epsilons = np.linspace(0, 1, num=steps + 1, dtype=npdtype)
        best = np.ones((N,), dtype=npdtype)

        for epsilon in epsilons:
            x = (1 - epsilon) * x0 + epsilon * random
            # TODO: due to limited floating point precision, clipping can be required
            logits = ep.astensor(self.model.forward(x.tensor))
            classes = logits.argmax(axis=-1)
            is_adv = (classes != labels).numpy()

            best = np.minimum(
                np.logical_not(is_adv).astype(npdtype)
                + is_adv.astype(npdtype) * epsilon,
                best,
            )

            if (best < 1).all():
                break

        best = ep.from_numpy(x0, best)
        best = atleast_kd(best, x0.ndim)
        x = (1 - best) * x0 + best * random
        return x.tensor
