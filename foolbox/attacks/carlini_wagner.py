import logging

import numpy as np

from .base import Attack
from .base import call_decorator


class CarliniWagnerAttack(Attack):
    """Implements Carlini & Wagner attack introduced in [1]_.
       Implements the l-2 norm version of the attack only,
       not the l0- oder l-infinity norms versions.
    References
    ----------
    .. [1] Nicholas Carlini & David Wagner,
            "Towards Evaluating the Robustness of Neural Networks",
            https://arxiv.org/abs/1608.04644
    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 confidence=5.0, learning_rate=1e-2, binary_search_steps=25,
                 max_iter=1000, initial_const=1e-3, decay=0.):

        """Simple and close to optimal gradient-based
        adversarial attack.
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
        confidence : int or float
            Confidence of adversarial examples: higher produces examples
            that are farther away, but more strongly classified as adversarial.
        learning_rate : float
            The learning rate for the attack algorithm. Smaller values
            produce better results but are slower to converge.
        binary_search_steps : int
            The number of times we perform binary search to
            find the optimal tradeoff-constant between distance and confidence.
        max_iter : int
            The maximum number of iterations. Larger values are more
            accurate; setting too small will require a large learning rate and
            will produce poor results.
        initial_const : float
            The initial tradeoff-constant to use to tune the relative
            importance of distance and confidence. If binary_search_steps is
            large, the initial constant is not important.
        decay : float
            Coefficient for learning rate decay.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            logging.fatal('Applied gradient-based attack to model that '
                          'does not provide gradients.')
            return

        if a.target_class() is None:
            logging.fatal('Carlini and Wagner is a targeted '
                          'adversarial attack.')
            return

        clip_min, clip_max = a.bounds()

        # for avoiding division by zero
        tanh_smoother = 1 - np.finfo(float).eps
        # if c exceeds this threshold, abort binary search
        const_upper_bound = 1e10
        const_lower_bound = 0

        perturbed_img = a.original_image.copy()
        const = initial_const

        # perform optimization in tanh space
        image_tanh = np.clip(perturbed_img, clip_min, clip_max)
        image_tanh = (image_tanh - clip_min) / (clip_max - clip_min)
        image_tanh = np.arctanh(((image_tanh * 2) - 1) * tanh_smoother)

        for _ in range(binary_search_steps):
            is_attack = False

            current_pertubation = np.zeros(image_tanh.shape)
            adam_optimizer = AdamOptimizer(current_pertubation.shape)

            for i in range(max_iter):
                # transform current adversarial from tanh to original space
                adversarial = image_tanh + current_pertubation
                adversarial = (np.tanh(adversarial) / tanh_smoother + 1) / 2
                adversarial = adversarial * (clip_max - clip_min) + clip_min
                adversarial = np.clip(adversarial, 0, 1)

                logits, squared_l2_dist, loss = CarliniWagnerAttack.loss(
                    a, a.original_image, adversarial,
                    a.target_class(), const, confidence)

                last_attack_success = loss - squared_l2_dist <= 0
                is_attack = is_attack or last_attack_success

                if last_attack_success:
                    break
                else:

                    target_one_hot = np.zeros(a.num_classes())
                    target_one_hot[a.target_class()] = 1

                    label_add = np.argmax(logits * (1 - target_one_hot))

                    gradient = a.gradient(adversarial, a.target_class())
                    gradient -= a.gradient(adversarial, label_add)
                    gradient *= const
                    gradient += 2 * (adversarial - a.original_image)
                    gradient *= (clip_max - clip_min)
                    squared_tanh = np.square(
                        np.tanh(image_tanh + current_pertubation))
                    gradient *= ((1 - squared_tanh) / (2 * tanh_smoother))

                    learning_rate *= (1. / (1. + decay * i))
                    current_pertubation = adam_optimizer(
                        gradient, learning_rate, current_pertubation)

            # find new const by using binary search
            if is_attack:
                const = (const_lower_bound + const) / 2
            else:
                const_old = const
                const = 2 * const
                const_lower_bound = const_old

            if const > const_upper_bound:
                break

    @staticmethod
    def loss(a, original_image, adversarial_image, target, const, confidence):
        dist = original_image - adversarial_image
        squared_l2_dist = np.sum(np.square(dist))
        logits, _, _ = a.predictions_and_gradient(adversarial_image)
        logits_target = logits[target]
        target_one_hot = np.zeros(a.num_classes())
        target_one_hot[target] = 1
        logits_other = np.max(logits * (1 - target_one_hot))

        loss = max(logits_other - logits_target + confidence, 0)

        return logits, squared_l2_dist, const * loss + squared_l2_dist


class AdamOptimizer:
    """Using the ADAM optimizer, as it is the most effective at quickly
    finding adversarial examples according to the paper [1]_.
    """
    def __init__(self, shape):
        """
        shape: (int, int)
            shape of the image
        """
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

    def __call__(self, gradient, learning_rate, current_pertubation,
                 beta1=0.9, beta2=0.999, epsilon=10**-8):
        """
        gradient: the gradient in the current iteration
        learning_rate: the learning rate in the current iteration
        current_pertubation: the image pertubation at the current iteration
        beta1: decay rate for calculating the exponentially
               decaying average of past gradients
        beta2: decay rate for calculating the exponentially
               decaying average of past squared gradients
        epsilon: to avoid division by zero
        :return: the newly calculated pertubation
        """
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient**2
        return current_pertubation - (learning_rate * self.m /
                                      (np.sqrt(self.v) + epsilon))
