import numpy as np

from .base import Attack


class SaliencyMapAttack(Attack):
    """Implements the Saliency Map Attack.

    The attack was introduced in [1]_.

    References
    ----------
    .. [1] The Limitations of Deep Learning in Adversarial Settings
           Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson,
           Z. Berkay Celik, Ananthram Swami
           https://arxiv.org/abs/1511.07528

    """

    def _apply(
            self,
            a,
            max_iter=2000,
            fast=True,
            theta=0.1,
            max_perturbations_per_pixel=7):

        """

        Parameters
        ----------
        theta : float
            perturbation per pixel relative to [min, max] range

        """

        # TODO: the original algorithm works on pixels across channels!

        target = a.target_class()
        if target is None:
            # TODO: choose one or more random targets
            # see lbfgs implementation
            return

        image = a.original_image()

        # the mask defines the search domain
        # each modified pixel with border value is set to zero in mask
        mask = np.ones_like(image)

        # count tracks how often each pixel was changed
        counts = np.zeros_like(image)

        # TODO: shouldn't this be without target
        labels = range(a.num_classes())

        perturbed = image

        max_, min_ = a.bounds()

        # TODO: stop if mask is all zero
        for step in range(max_iter):
            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                return

            # get pixel location with highest influence on class
            idx, p_sign = self._saliency_map(
                a, perturbed, target, labels, mask, fast=fast)

            # apply perturbation
            perturbed[idx] += -p_sign * theta * (max_ - min_)

            # tracks number of updates for each pixel
            counts[idx] += 1

            # remove pixel from search domain if it hits the bound
            if perturbed[idx] <= min_ or perturbed[idx] >= max_:
                mask[idx] = 0

            # remove pixel if it was changed too often
            if counts[idx] >= max_perturbations_per_pixel:
                mask[idx] = 0

            perturbed = np.clip(perturbed, min_, max_)

    def _saliency_map(self, a, image, target, labels, mask, fast=False):
        """Implements Algorithm 3 in manuscript

        """

        # pixel influence on target class
        alphas = a.gradient(image, target) * mask

        # pixel influence on sum of residual classes
        # (don't evaluate if fast == True)
        if fast:
            betas = -np.ones_like(alphas)
        else:
            betas = np.sum([
                a.gradient(image, label) * mask - alphas
                for label in labels], 0)

        # compute saliency map
        # (take into account both pos. & neg. perturbations)
        salmap = np.abs(alphas) * np.abs(betas) * np.sign(alphas * betas)

        # find optimal pixel & direction of perturbation
        idx = np.argmin(salmap)
        idx = np.unravel_index(idx, mask.shape)
        pix_sign = np.sign(alphas)[idx]

        return idx, pix_sign
