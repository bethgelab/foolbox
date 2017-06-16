import logging
import random

import numpy as np

from .base import Attack
from ..utils import crossentropy


class DeepFoolAttack(Attack):
    """Simple and accurate adversarial attack.

    Implementes DeepFool introduced in [1]_.

    References
    ----------
    .. [1] DeepFool: a simple and accurate method to fool deep neural networks
           Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard
           https://arxiv.org/abs/1511.04599
    """

    def _apply(self, a, steps=100, subsample=False):
        if not a.has_gradient():
            return

        if a.target_class() is not None:
            logging.warn('Targeted adversarials not supported by DeepFool.')
            return

        if subsample:
            logging.warn('Performs subsampling, results will be suboptimal.')

        label = a.original_class

        def get_residual_labels(logits):
            """Get all labels with p < p[target]"""
            n = logits.shape[0]
            return [
                k for k in range(n)
                if logits[k] < logits[label]]

        perturbed = a.original_image
        min_, max_ = a.bounds()

        for step in range(steps):
            logits, grad, is_adv = a.predictions_and_gradient(perturbed)
            if is_adv:
                return

            # correspondance to algorithm 2 in [1]_:
            #
            # loss corresponds to f (in the paper: negative cross-entropy)
            # grad corresponds to -df/dx (gradient of cross-entropy)

            loss = -crossentropy(logits=logits, label=label)

            residual_labels = get_residual_labels(logits)
            if subsample:
                assert isinstance(subsample, int)
                random.shuffle(residual_labels)
                residual_labels = residual_labels[:subsample]

            losses = [
                -crossentropy(logits=logits, label=k)
                for k in residual_labels]
            grads = [a.gradient(perturbed, label=k) for k in residual_labels]

            # compute optimal direction (and loss difference)
            # pairwise between each label and the target
            diffs = [(l - loss, g - grad) for l, g in zip(losses, grads)]

            # calculate distances
            distances = [abs(dl) / (np.linalg.norm(dg) + 1e-8) for dl, dg in diffs]  # noqa: E501

            # choose optimal one
            optimal = np.argmin(distances)
            df, dg = diffs[optimal]

            # apply perturbation
            # the (-dg) corrects the sign, gradient here is -gradient of paper
            perturbation = abs(df) / (np.linalg.norm(dg) + 1e-8)**2 * (-dg)
            perturbed = perturbed + 1.05 * perturbation
            perturbed = np.clip(perturbed, min_, max_)

        a.predictions(perturbed)  # to find an adversarial in the last step


DeepFool = DeepFoolAttack
