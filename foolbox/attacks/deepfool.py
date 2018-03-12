import logging

import numpy as np

from .base import Attack
from ..utils import crossentropy
from ..distances import MeanSquaredDistance
from ..distances import Linfinity


class DeepFoolAttack(Attack):
    """Simple and accurate adversarial attack.

    Implementes DeepFool introduced in [1]_.

    References
    ----------
    .. [1] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
           "DeepFool: a simple and accurate method to fool deep neural
           networks", https://arxiv.org/abs/1511.04599

    """

    def _apply(self, a, steps=100, subsample=10, p=None):
        if not a.has_gradient():
            return

        if a.target_class() is not None:
            logging.fatal('DeepFool is an untargeted adversarial attack.')
            return

        if p is None:
            # set norm to optimize based on the distance measure
            if a._distance == MeanSquaredDistance:
                p = 2
            elif a._distance == Linfinity:
                p = np.inf
            else:
                raise NotImplementedError('Please choose a distance measure'
                                          ' for which DeepFool is implemented'
                                          ' or specify manually which norm'
                                          ' to optimize.')

        if not (1 <= p <= np.inf):
            raise ValueError

        if p not in [2, np.inf]:
            raise NotImplementedError

        label = a.original_class

        # define labels
        logits, _ = a.predictions(a.original_image)
        labels = np.argsort(logits)[::-1]
        if subsample:
            # choose the top-k classes
            logging.info('Only testing the top-{} classes'.format(subsample))
            assert isinstance(subsample, int)
            labels = labels[:subsample]

        def get_residual_labels(logits):
            """Get all labels with p < p[original_class]"""
            return [
                k for k in labels
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

            # instead of using the logits and the gradient of the logits,
            # we use a numerically stable implementation of the cross-entropy
            # and expect that the deep learning frameworks also use such a
            # stable implemenation to calculate the gradient
            losses = [
                -crossentropy(logits=logits, label=k)
                for k in residual_labels]
            grads = [a.gradient(perturbed, label=k) for k in residual_labels]

            # compute optimal direction (and loss difference)
            # pairwise between each label and the target
            diffs = [(l - loss, g - grad) for l, g in zip(losses, grads)]

            # calculate distances
            if p == 2:
                distances = [abs(dl) / (np.linalg.norm(dg) + 1e-8)
                             for dl, dg in diffs]
            elif p == np.inf:
                distances = [abs(dl) / (np.sum(np.abs(dg)) + 1e-8)
                             for dl, dg in diffs]
            else:  # pragma: no cover
                assert False

            # choose optimal one
            optimal = np.argmin(distances)
            df, dg = diffs[optimal]

            # apply perturbation
            # the (-dg) corrects the sign, gradient here is -gradient of paper
            if p == 2:
                perturbation = abs(df) / (np.linalg.norm(dg) + 1e-8)**2 * (-dg)
            elif p == np.inf:
                perturbation = abs(df) / (np.sum(np.abs(dg)) + 1e-8) \
                    * np.sign(-dg)
            else:  # pragma: no cover
                assert False

            # the original implementation accumulates the perturbations
            # and only adds the overshoot when adding the accumulated
            # perturbation to the original image; we apply the overshoot
            # to each perturbation (step)
            perturbed = perturbed + 1.05 * perturbation
            perturbed = np.clip(perturbed, min_, max_)

        a.predictions(perturbed)  # to find an adversarial in the last step


class DeepFoolL2Attack(DeepFoolAttack):
    def _apply(self, a, steps=100, subsample=10):
        super(DeepFoolL2Attack, self)._apply(
            a, steps=steps, subsample=subsample, p=2)


class DeepFoolLinfinityAttack(DeepFoolAttack):
    def _apply(self, a, steps=100, subsample=10):
        super(DeepFoolLinfinityAttack, self)._apply(
            a, steps=steps, subsample=subsample, p=np.inf)
