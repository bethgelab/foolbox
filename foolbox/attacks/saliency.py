import logging
import random

import numpy as np

from .base import Attack
from .gradient import GradientAttack


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
            num_random_targets=0,
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

        original_class = a.original_class

        target_class = a.target_class()
        if target_class is None:
            if num_random_targets == 0:
                gradient_attack = GradientAttack()
                gradient_attack(a)
                adv_img = a.image
                if adv_img is None:  # pragma: no coverage
                    # using GradientAttack did not work,
                    # falling back to random target
                    num_random_targets = 1
                    logging.info('Using GradientAttack to determine a target class failed, falling back to a random target class')  # noqa: E501
                else:
                    logits, _ = a.predictions(adv_img)
                    target_class = np.argmax(logits)
                    target_classes = [target_class]
                    logging.info('Determined a target class using the GradientAttack: {}'.format(target_class))  # noqa: E501
            else:  # pragma: no coverage
                num_random_targets = 1

            if num_random_targets > 0:

                # draw num_random_targets random classes all of which are
                # different and not the original class

                num_classes = a.num_classes()
                assert num_random_targets <= num_classes - 1

                # sample one more than necessary
                # remove original class from samples
                # should be more efficient than other approaches, see
                # https://github.com/numpy/numpy/issues/2764
                target_classes = random.sample(
                    range(num_classes), num_random_targets + 1)
                target_classes = [t for t in target_classes if t != original_class]  # noqa: E501
                target_classes = target_classes[:num_random_targets]

                str_target_classes = [str(t) for t in target_classes]
                logging.info('Random target classes: {}'.format(', '.join(str_target_classes)))  # noqa: E501
        else:
            target_classes = [target_class]

        for target in target_classes:

            image = a.original_image

            # the mask defines the search domain
            # each modified pixel with border value is set to zero in mask
            mask = np.ones_like(image)

            # count tracks how often each pixel was changed
            counts = np.zeros_like(image)

            # TODO: shouldn't this be without target
            labels = range(a.num_classes())

            perturbed = image.copy()

            min_, max_ = a.bounds()

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
