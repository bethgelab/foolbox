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
            location.insert(channel_axis, None)
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
    .. [1] Simple Black-Box Adversarial Perturbations for Deep Networks
           Nina Narodytska, Shiva Prasad Kasiviswanathan
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
            locations = np.random.permutation(h * w)[:int(0.1 * h * w)]
            p_x = locations % w
            p_y = locations // w
            pxy = list(zip(p_x, p_y))
            pxy = np.array(pxy)
            return pxy

        def pert(Ii, p, x, y):
            I = Ii.copy()
            location = [x, y]
            location.insert(channel_axis, None)
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
            L = [pert(Ii, p, x, y) for x, y in PxPy]

            # TODO: use batch predictions
            def score(It):
                logits, _ = a.predictions(unnormalize(It), strict=False)
                probs = softmax(logits)
                return probs[cI]

            scores = [score(It) for It in L]

            indices = np.argsort(scores)[::-1][:t]

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
            if is_adv:
                return

            # Update a neighborhood of pixel locations for the next round
            PxPy = [
                (x, y)
                for _a, _b in PxPy_star
                for x in range(_a - d, _a + d + 1)
                for y in range(_b - d, _b + d + 1)]
            PxPy = [(x, y) for x, y in PxPy if 0 <= x < w and 0 <= y < h]
            PxPy = np.array(PxPy)
