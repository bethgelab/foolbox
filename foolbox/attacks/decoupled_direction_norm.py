import math
import numpy as np
import logging

from .base import Attack
from .base import generator_decorator


class DecoupledDirectionNormL2Attack(Attack):
    """The Decoupled Direction and Norm L2 adversarial attack from [1]_.

    References
    ----------
    .. [1] Jérôme Rony, Luiz G. Hafemann, Luiz S. Oliveira, Ismail Ben Ayed,
    Robert Sabourin, Eric Granger, "Decoupling Direction and Norm for Efficient
    Gradient-Based L2 Adversarial Attacks and Defenses",
    https://arxiv.org/abs/1811.09600

    """

    @generator_decorator
    def as_generator(
        self, a, steps=100, gamma=0.05, initial_norm=1, quantize=True, levels=256
    ):
        """The Decoupled Direction and Norm L2 adversarial attack.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        steps : int
            Number of steps for the optimization.
        gamma : float, optional
            Factor by which the norm will be modified.
            new_norm = norm * (1 + or - gamma).
        init_norm : float, optional
            Initial value for the norm.
        quantize : bool, optional
            If True, the returned adversarials will have quantized values to
            the specified number of levels.
        levels : int, optional
            Number of levels to use for quantization
            (e.g. 256 for 8 bit images).

        """

        if not a.has_gradient():
            logging.fatal(
                "Applied gradient-based attack to model that "
                "does not provide gradients."
            )
            return

        min_, max_ = a.bounds()
        s = max_ - min_
        if a.target_class is not None:
            multiplier = -1
            attack_class = a.target_class
        else:
            multiplier = 1
            attack_class = a.original_class
        norm = initial_norm
        unperturbed = a.unperturbed
        perturbation = np.zeros_like(unperturbed)

        for i in range(steps):
            logits, grad, is_adv = yield from a.forward_and_gradient_one(
                unperturbed + perturbation, attack_class, strict=True
            )

            # renorm gradient and handle 0-norm gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm == 0:  # pragma: no cover
                grad = np.random.normal(size=grad.shape)
                grad_norm = np.linalg.norm(grad)
            grad *= s / grad_norm

            # udpate perturbation
            lr = cosine_learning_rate(i, steps, 1.0, 0.01)
            perturbation += lr * multiplier * grad

            # update norm value and renorm perturbation accordingly
            norm *= 1 - (2 * is_adv - 1) * gamma
            perturbation *= s * norm / np.linalg.norm(perturbation)
            if quantize:
                perturbation = (perturbation - min_) / s
                perturbation = np.round(perturbation * (levels - 1))
                perturbation /= levels - 1
                perturbation = perturbation * s + min_
            perturbation = np.clip(perturbation, min_ - unperturbed, max_ - unperturbed)


def cosine_learning_rate(current_step, max_steps, init_lr, final_lr):
    """Cosine annealing schedule for learning rate.

    Parameters
    ----------
    current_step : int
        Current step in the optimization
    max_steps : int
        Total number of steps of the optimization.
    init_lr : float
        Initial learning rate.
    final_lr : float
        Final learning rate.

    Returns
    -------
    float
        The current learning rate.

    """
    alpha = (1 + math.cos(math.pi * current_step / max_steps)) / 2
    return final_lr + alpha * (init_lr - final_lr)
