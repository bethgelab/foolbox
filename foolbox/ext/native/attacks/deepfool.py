import eagerpy as ep
import numpy as np
import logging
from abc import ABC
from abc import abstractmethod

from ..utils import flatten
from ..utils import atleast_kd


class DeepFoolAttack(ABC):
    """A simple and fast gradient-based adversarial attack.

    Implementes DeepFool introduced in [1]_.

    References
    ----------
    .. [1] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
           "DeepFool: a simple and accurate method to fool deep neural
           networks", https://arxiv.org/abs/1511.04599

    """

    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        inputs,
        labels,
        *,
        p,
        candidates=10,
        overshoot=0.02,
        steps=50,
        loss="logits",
    ):
        """
        Parameters
        ----------
        p : int or float
            Lp-norm that should be minimzed, must be 2 or np.inf.
        candidates : int
            Limit on the number of the most likely classes that should
            be considered. A small value is usually sufficient and much
            faster.
        overshoot : float
        steps : int
            Maximum number of steps to perform.
        """

        if not (1 <= p <= np.inf):
            raise ValueError
        if p not in [2, np.inf]:
            raise NotImplementedError

        min_, max_ = self.model.bounds()

        inputs = ep.astensor(inputs)
        labels = ep.astensor(labels)

        N = len(inputs)

        logits = ep.astensor(self.model.forward(inputs.tensor))
        candidates = min(candidates, logits.shape[-1])
        classes = logits.argsort(axis=-1).flip(axis=-1)
        if candidates:
            assert candidates >= 2
            logging.info(f"Only testing the top-{candidates} classes")
            classes = classes[:, :candidates]

        i0 = classes[:, 0]
        rows = ep.arange(inputs, N)

        if loss == "logits":

            def loss_fun(x: ep.Tensor, k: int) -> ep.Tensor:
                logits = ep.astensor(self.model.forward(x.tensor))
                ik = classes[:, k]
                l0 = logits[rows, i0]
                lk = logits[rows, ik]
                loss = lk - l0
                return loss.sum(), (loss, logits)

        elif loss == "crossentropy":

            def loss_fun(x: ep.Tensor, k: int) -> ep.Tensor:
                logits = ep.astensor(self.model.forward(x.tensor))
                ik = classes[:, k]
                l0 = -ep.crossentropy(logits, i0)
                lk = -ep.crossentropy(logits, ik)
                loss = lk - l0
                return loss.sum(), (loss, logits)

        else:
            raise ValueError(
                f"expected loss to be 'logits' or 'crossentropy', got '{loss}'"
            )

        loss_aux_and_grad = ep.value_and_grad_fn(inputs, loss_fun, has_aux=True)

        x = x0 = inputs
        p_total = ep.zeros_like(x)
        for step in range(steps):
            # let's first get the logits using k = 1 to see if we are done
            diffs = [loss_aux_and_grad(x, 1)]
            _, (_, logits), _ = diffs[0]
            is_adv = logits.argmax(axis=-1) != labels
            if is_adv.all():
                break
            # then run all the other k's as well
            # we could avoid repeated forward passes and only repeat
            # the backward pass, but this cannot currently be done in eagerpy
            diffs += [loss_aux_and_grad(x, k) for k in range(2, candidates)]

            # we don't need the logits
            diffs = [(losses, grad) for _, (losses, _), grad in diffs]
            losses = ep.stack([l for l, _ in diffs], axis=1)
            grads = ep.stack([g for _, g in diffs], axis=1)
            assert losses.shape == (N, candidates - 1)
            assert grads.shape == (N, candidates - 1) + x0.shape[1:]

            # calculate the distances
            distances = self.get_distances(losses, grads)
            assert distances.shape == (N, candidates - 1)

            # determine the best directions
            best = distances.argmin(axis=1)
            distances = distances[rows, best]
            losses = losses[rows, best]
            grads = grads[rows, best]
            assert distances.shape == (N,)
            assert losses.shape == (N,)
            assert grads.shape == x0.shape

            # apply perturbation
            distances = distances + 1e-4  # for numerical stability
            p_step = self.get_perturbations(distances, grads)
            assert p_step.shape == x0.shape

            p_total += p_step
            # don't do anything for those that are already adversarial
            x = ep.where(
                atleast_kd(is_adv, x.ndim), x, x0 + (1.0 + overshoot) * p_total
            )
            x = ep.clip(x, min_, max_)

        return x.tensor

    @abstractmethod
    def get_distances(self, losses, grads):
        raise NotImplementedError

    @abstractmethod
    def get_perturbations(self, distances, grads):
        raise NotImplementedError


class L2DeepFoolAttack(DeepFoolAttack):
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, p=2, **kwargs)

    def get_distances(self, losses, grads):
        return abs(losses) / (
            flatten(grads, keep=2).square().sum(axis=-1).sqrt() + 1e-8
        )

    def get_perturbations(self, distances, grads):
        return (
            atleast_kd(
                distances / (flatten(grads).square().sum(axis=-1).sqrt() + 1e-8),
                grads.ndim,
            )
            * grads
        )


class LinfDeepFoolAttack(DeepFoolAttack):
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, p=np.inf, **kwargs)

    def get_distances(self, losses, grads):
        return abs(losses) / (flatten(grads, keep=2).abs().sum(axis=-1) + 1e-8)

    def get_perturbations(self, distances, grads):
        return atleast_kd(distances, grads.ndim) * grads.sign()
