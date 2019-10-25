import numpy as np

from .base import Attack
from .base import call_decorator
from ...utils import softmax
from ... import nprng


class SinglePixelAttack(Attack):
    """Perturbs just a single pixel and sets it to the min or max."""

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True, max_pixels=1000):

        """Perturbs just a single pixel and sets it to the min or max.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified input. If it is a
            numpy array, label must be passed as well. If it is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original input. Must be passed
            if input is a numpy array, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        max_pixels : int
            Maximum number of pixels to try.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        channel_axis = a.channel_axis(batch=False)
        axes = [i for i in range(a.unperturbed.ndim) if i != channel_axis]
        assert len(axes) == 2
        h = a.unperturbed.shape[axes[0]]
        w = a.unperturbed.shape[axes[1]]

        min_, max_ = a.bounds()

        pixels = nprng.permutation(h * w)
        pixels = pixels[:max_pixels]
        for i, pixel in enumerate(pixels):
            x = pixel % w
            y = pixel // w

            location = [x, y]
            location.insert(channel_axis, slice(None))
            location = tuple(location)

            for value in [min_, max_]:
                perturbed = a.unperturbed.copy()
                perturbed[location] = value

                _, is_adv = a.forward_one(perturbed)
                if is_adv:
                    return


class LocalSearchAttack(Attack):
    """A black-box attack based on the idea of greedy local search.

    This implementation is based on the algorithm in [1]_.

    References
    ----------
    .. [1] Nina Narodytska, Shiva Prasad Kasiviswanathan, "Simple
           Black-Box Adversarial Perturbations for Deep Networks",
           https://arxiv.org/abs/1612.06299

    """

    @call_decorator
    def __call__(
        self, input_or_adv, label=None, unpack=True, r=1.5, p=10.0, d=5, t=5, R=150
    ):

        """A black-box attack based on the idea of greedy local search.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified input. If it is a
            numpy array, label must be passed as well. If it is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original input. Must be passed
            if input is a numpy array, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        r : float
            Perturbation parameter that controls the cyclic perturbation;
            must be in [0, 2]
        p : float
            Perturbation parameter that controls the pixel sensitivity
            estimation
        d : int
            The half side length of the neighborhood square
        t : int
            The number of pixels perturbed at each round
        R : int
            An upper bound on the number of iterations

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        # TODO: incorporate the modifications mentioned in the manuscript
        # under "Implementing Algorithm LocSearchAdv"

        assert 0 <= r <= 2

        if a.target_class is not None:
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

        Im = a.unperturbed
        Im, LB, UB = normalize(Im)

        cI = a.original_class

        channel_axis = a.channel_axis(batch=False)
        axes = [i for i in range(Im.ndim) if i != channel_axis]
        assert len(axes) == 2
        h = Im.shape[axes[0]]
        w = Im.shape[axes[1]]
        channels = Im.shape[channel_axis]

        def random_locations():
            n = int(0.1 * h * w)
            n = min(n, 128)
            locations = nprng.permutation(h * w)[:n]
            p_x = locations % w
            p_y = locations // w
            pxy = list(zip(p_x, p_y))
            pxy = np.array(pxy)
            return pxy

        def pert(Ii, p, x, y):
            Im = Ii.copy()
            location = [x, y]
            location.insert(channel_axis, slice(None))
            location = tuple(location)
            Im[location] = p * np.sign(Im[location])
            return Im

        def cyclic(r, Ibxy):
            result = r * Ibxy
            if result < LB:
                result = result + (UB - LB)
            elif result > UB:
                result = result - (UB - LB)
            assert LB <= result <= UB
            return result

        Ii = Im
        PxPy = random_locations()

        for _ in range(R):
            # Computing the function g using the neighborhood
            # IMPORTANT: random subset for efficiency
            PxPy = PxPy[nprng.permutation(len(PxPy))[:128]]
            L = [pert(Ii, p, x, y) for x, y in PxPy]

            def score(Its):
                Its = np.stack(Its)
                Its = unnormalize(Its)
                batch_logits, _ = a.forward(Its, strict=False)
                scores = [softmax(logits)[cI] for logits in batch_logits]
                return scores

            scores = score(L)

            indices = np.argsort(scores)[-t:]

            PxPy_star = PxPy[indices]

            # Generation of new perturbed input Ii
            for x, y in PxPy_star:
                for b in range(channels):
                    location = [x, y]
                    location.insert(channel_axis, b)
                    location = tuple(location)
                    Ii[location] = cyclic(r, Ii[location])

            # Check whether the perturbed input Ii is an adversarial input
            _, is_adv = a.forward_one(unnormalize(Ii))
            if is_adv:  # pragma: no cover
                return

            # Update a neighborhood of pixel locations for the next round
            PxPy = [
                (x, y)
                for _a, _b in PxPy_star
                for x in range(_a - d, _a + d + 1)
                for y in range(_b - d, _b + d + 1)
            ]
            PxPy = [(x, y) for x, y in PxPy if 0 <= x < w and 0 <= y < h]
            PxPy = list(set(PxPy))
            PxPy = np.array(PxPy)
