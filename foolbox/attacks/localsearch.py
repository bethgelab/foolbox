from __future__ import division
import numpy as np

from .base import Attack
from ..utils import softmax


class SinglePixelAttack(Attack):
    """Perturbs just a single pixel and sets it to the min or max.

    """

    def _apply(self, a, max_pixels=1000):
        channel_axis = a.channel_axis(batch=False)
        image = a.original_image
        axes = [i for i in range(image.ndim) if i != channel_axis]
        assert len(axes) == 2
        h = image.shape[axes[0]]
        w = image.shape[axes[1]]

        min_, max_ = a.bounds()

        pixels = np.random.permutation(h * w)
        pixels = pixels[:max_pixels]
        for i, pixel in enumerate(pixels):
            x = pixel % w
            y = pixel // w

            location = [x, y]
            location.insert(channel_axis, slice(None))
            location = tuple(location)

            for value in [min_, max_]:
                perturbed = image.copy()
                perturbed[location] = value

                _, is_adv = a.predictions(perturbed)
                if is_adv:
                    return


class LocalSearchAttack(Attack):
    """A black-box attack based on the idea of greedy local search.

    This implementation is based on the algorithm in [1]_.

    References
    ----------
    .. [1] Nina Narodytska, Shiva Prasad Kasiviswanathan, "Simple
           Black-Box Adversarial Perturbations for Deep Networks",
           https://arxiv.org/pdf/1612.06299.pdf

    """

    def _apply(self, a, r=1.5, p=10., d=5, t=5, R=150):
        # TODO: incorporate the modifications mentioned in the manuscript
        # under "Implementing Algorithm LocSearchAdv"

        if a.target_class() is not None:
            # TODO: check if this algorithm can be used as a targeted attack
            return

        def normalize(im):
            min_, max_ = a.bounds()

            im = im - (min_ + max_) / 2
            im = im / (max_ - min_)

            LB = -1 / 2
            UB = 1 / 2
            return im, LB, UB

        def unnormalize(im):
            min_, max_ = a.bounds()

            im = im * (max_ - min_)
            im = im + (min_ + max_) / 2
            return im

        I = a.original_image
        I, LB, UB = normalize(I)

        cI = a.original_class

        channel_axis = a.channel_axis(batch=False)
        axes = [i for i in range(I.ndim) if i != channel_axis]
        assert len(axes) == 2
        h = I.shape[axes[0]]
        w = I.shape[axes[1]]
        channels = I.shape[channel_axis]

        def random_locations():
            n = int(0.1 * h * w)
            n = min(n, 128)
            locations = np.random.permutation(h * w)[:n]
            p_x = locations % w
            p_y = locations // w
            pxy = list(zip(p_x, p_y))
            pxy = np.array(pxy)
            return pxy

        def pert(Ii, p, x, y):
            I = Ii.copy()
            location = [x, y]
            location.insert(channel_axis, slice(None))
            location = tuple(location)
            I[location] = p * np.sign(I[location])
            return I

        def cyclic(r, Ibxy):
            result = r * Ibxy
            if result < LB:
                result = result + (UB - LB)
            elif result > UB:
                result = result - (UB - LB)
            assert LB <= result <= UB
            return result

        Ii = I
        PxPy = random_locations()

        for _ in range(R):
            # Computing the function g using the neighborhood
            # IMPORTANT: random subset for efficiency
            PxPy = PxPy[np.random.permutation(len(PxPy))[:128]]
            L = [pert(Ii, p, x, y) for x, y in PxPy]

            def score(Its):
                Its = np.stack(Its)
                Its = unnormalize(Its)
                batch_logits, _ = a.batch_predictions(Its, strict=False)
                scores = [softmax(logits)[cI] for logits in batch_logits]
                return scores

            scores = score(L)

            indices = np.argsort(scores)[:t]

            PxPy_star = PxPy[indices]

            # Generation of new perturbed image Ii
            for x, y in PxPy_star:
                for b in range(channels):
                    location = [x, y]
                    location.insert(channel_axis, b)
                    location = tuple(location)
                    Ii[location] = cyclic(r, Ii[location])

            # Check whether the perturbed image Ii is an adversarial image
            _, is_adv = a.predictions(unnormalize(Ii))
            if is_adv:  # pragma: no cover
                return

            # Update a neighborhood of pixel locations for the next round
            PxPy = [
                (x, y)
                for _a, _b in PxPy_star
                for x in range(_a - d, _a + d + 1)
                for y in range(_b - d, _b + d + 1)]
            PxPy = [(x, y) for x, y in PxPy if 0 <= x < w and 0 <= y < h]
            PxPy = np.array(PxPy)
