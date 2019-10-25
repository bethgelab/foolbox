import numpy as np
from collections import Iterable
import logging

from .base import Attack
from .base import generator_decorator
from ..utils import softmax


class VirtualAdversarialAttack(Attack):
    """Calculate an untargeted adversarial perturbation by performing a
    approximated second order optimization step on the KL divergence between
    the unperturbed predictions and the predictions for the adversarial
    perturbation. This attack was introduced in [1]_.

    References
    ----------
    .. [1] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae,
           Shin Ishii,
           "Distributional Smoothing with Virtual Adversarial Training",
           https://arxiv.org/abs/1507.00677
    """

    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.sqrt(np.mean(np.square(perturbation)))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor

    @generator_decorator
    def as_generator(self, a, xi=1e-5, iterations=1, epsilons=1000, max_epsilon=0.3):
        """

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        xi : float
            The finite difference size for performing the power method.
        iterations : int
            Number of iterations to perform power method to search for second
            order perturbation of KL divergence.
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the direction of the sign of
            the gradient or number of step sizes between 0 and max_epsilon
            that should be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.
        """

        assert (
            a.target_class is None
        ), "Virtual Adversarial is an untargeted adversarial attack."

        yield from self._run(
            a, xi=xi, iterations=iterations, epsilons=epsilons, max_epsilon=max_epsilon
        )

    def _run(self, a, xi, iterations, epsilons, max_epsilon):
        if not a.has_gradient():
            return

        x = a.unperturbed
        min_, max_ = a.bounds()

        logits, _ = yield from a.forward_one(x)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, max_epsilon, num=epsilons + 1)[1:]
            decrease_if_first = True
        else:
            decrease_if_first = False

        for _ in range(2):  # to repeat with decreased epsilons if necessary
            for i, epsilon in enumerate(epsilons):
                # start with random vector as search vector
                d = np.random.normal(0.0, 1.0, size=x.shape).astype(x.dtype)
                for it in range(iterations):
                    # normalize proposal to be unit vector
                    d = xi * d / np.sqrt((d ** 2).sum())

                    logits_d, _ = yield from a.forward_one(x + d, strict=False)

                    dl_dp = softmax(logits) - softmax(logits_d)

                    # d = dl_dd = dl_dp * dp_dd
                    # use gradient of KL divergence as new search vector
                    d = yield from a.backward_one(gradient=dl_dp, x=x + d, strict=False)
                    d = (max_ - min_) * d

                    if np.allclose(np.sqrt((d ** 2).sum()), 0, atol=1e-16):
                        raise RuntimeError(
                            "Gradient vanished; this can happen if xi is too small."
                        )

                delta = d / np.sqrt((d ** 2).mean())

                perturbed = x + self._clip_perturbation(a, delta, epsilon)
                perturbed = np.clip(perturbed, min_, max_)

                _, is_adversarial = yield from a.forward_one(perturbed)
                if is_adversarial:
                    if decrease_if_first and i < 20:
                        logging.info("repeating attack with smaller epsilons")
                        break
                    return

            max_epsilon = epsilons[i]
            epsilons = np.linspace(0, max_epsilon, num=20 + 1)[1:]
