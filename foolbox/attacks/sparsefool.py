import logging

import numpy as np

from .base import Attack
from .base import call_decorator
from ..utils import crossentropy


class SparseFoolAttack(Attack):
    """A geometry-inspired and fast attack for computing
    sparse adversarial perturbations.

    Implements SparseFool introduced in [1]_.
    The official code is provided in [3]_.

    References
    ----------
    .. [1] Apostolos Modas, Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard,
           "SparseFool: a few pixels make a big difference",
           https://arxiv.org/abs/1811.02248

    .. [2] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
           "DeepFool: a simple and accurate method to fool deep neural
           networks", https://arxiv.org/abs/1511.04599

    .. [3] https://github.com/LTS4/SparseFool

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 steps=30, lambda_=1., subsample=10):

        """A geometry-inspired and fast attack for computing
        sparse adversarial perturbations.

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
            Maximum number of steps to perform.
        lambda_ : float
            Pushes the approximated decision boundary deeper into the
            classification region of the fooling class.
        subsample : int
            Limit on the number of the most likely classes that should
            be considered when approximating the decision boundary. A small
            value is usually sufficient and much faster.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            return

        if a.target_class() is not None:
            logging.fatal('SparseFool is an untargeted adversarial attack.')
            return

        _label = a.original_class

        min_, max_ = a.bounds()
        perturbed = a.unperturbed
        for step in range(steps):

            logits, grad, is_adv = a.forward_and_gradient_one(perturbed)
            if is_adv:
                return

            # Approximate the decision boundary as an affine hyperplane. To do
            # so, we need a data point that lies on the decision boundary, and
            # the normal of the decision boundary at that point. The final
            # approximation is done using an overshooted version of the
            # boundary point by a factor lambda_.
            boundary_point, boundary_normal = \
                self.boundary_approximation_deepfool(a, perturbed, subsample,
                                                     _label, lambda_)

            if boundary_point is None:
                logging.fatal('SparseFool fails to find an adversarial.')
                return

            # Compute the l1 perturbation between the current adversarial point
            # and the approximated hyperplane
            perturbation = self.l1_linear_solver(perturbed, boundary_point,
                                                 boundary_normal, min_, max_)

            # Update the current iterate
            perturbed = np.clip(perturbed + 1.02 * perturbation, min_, max_)

        a.forward_one(perturbed)  # to find an adversarial in the last step

    @classmethod
    def boundary_approximation_deepfool(cls, a, initial_point, subsample,
                                        label, lambda_, steps=100):

        """Approximates the decision boundary as an affine hyperplane.
        The approximation is done using a slightly modified version of
        the unconstrained DeepFool introduced in [2]_.

        Parameters
        ----------
        a : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        initial_point : `numpy.ndarray`
            The initial point that we want to move towards the decision
            boundary of the fooling class.
        subsample : int
            Limit on the number of the most likely classes that should
            be considered. A small value is usually sufficient and much
            faster.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        lambda_ : float
            Specifies the factor by which the boundary point is pushed
            further into the classification region of the fooling class.
        steps : int
            Maximum number of steps to perform.

        """

        # define labels
        logits, _ = a.forward_one(initial_point)
        labels = np.argsort(logits)[::-1]
        if subsample:
            # choose the top-k classes
            logging.info('Only testing the top-{} classes'.format(subsample))
            assert isinstance(subsample, int)
            labels = labels[:subsample]

        def get_residual_labels(logs):
            """Get all labels with p < p[original_class]"""
            return [
                k for k in labels
                if logs[k] < logs[label]]

        total_perturbation = 0.
        for step in range(steps):

            # Update the boundary point
            boundary_point = initial_point + total_perturbation

            logits, grad, is_adv = \
                a.forward_and_gradient_one(
                    initial_point + 1.02 * total_perturbation, strict=False)

            if is_adv:

                # Get fooling label
                labels = np.argsort(logits)[::-1]
                fooling_label = np.argmax(logits)

                # Compute the gradients at the boundary point
                grad_fool = a.gradient_one(boundary_point, label=fooling_label, strict=False)
                grad_true = a.gradient_one(boundary_point, label=label, strict=False)
                grad_diff = grad_fool - grad_true

                # Compute the normal and correct the sign of the gradient
                normal = (-grad_diff) / np.linalg.norm(grad_diff)

                # Return the overshooted boundary point and the normal to the
                # decision boundary
                return initial_point + lambda_ * total_perturbation, normal

            # correspondence to algorithm 2 in [2]_:
            #
            # loss corresponds to f (in the paper: negative cross-entropy)
            # grad corresponds to -df/dx (gradient of cross-entropy)

            loss = -crossentropy(logits=logits, label=label)

            residual_labels = get_residual_labels(logits)

            # sanity check: original label has zero probability or is not in
            # the labels
            if len(residual_labels) == 0:
                logging.fatal('No label with p < p[original_class]')

                return None, _

            # instead of using the logits and the gradient of the logits,
            # we use a numerically stable implementation of the cross-entropy
            # and expect that the deep learning frameworks also use such a
            # stable implementation to calculate the gradient
            losses = [-crossentropy(logits=logits, label=k)
                      for k in residual_labels]
            grads = [a.gradient_one(boundary_point, label=k, strict=False)
                     for k in residual_labels]

            # compute optimal direction (and loss difference)
            # pairwise between each label and the target
            diffs = [(l - loss, g - grad) for l, g in zip(losses, grads)]

            # calculate distances
            distances = [abs(dl) / (np.linalg.norm(dg) + 1e-8)
                         for dl, dg in diffs]

            # choose optimal distance
            optimal = np.argmin(distances)
            df, dg = diffs[optimal]

            # apply perturbation
            # the (-dg) corrects the sign, gradient here is -gradient of paper
            perturbation = abs(df) / (np.linalg.norm(dg) + 1e-8) ** 2 * (-dg)
            total_perturbation = total_perturbation + perturbation

    @classmethod
    def l1_linear_solver(cls, initial_point, boundary_point,
                         normal, min_, max_):

        """Computes the L1 solution (perturbation) to the linearized problem.
        It corresponds to algorithm 1 in [1]_.

        Parameters
        ----------
        initial_point : `numpy.ndarray`
            The initial point for which we seek the L1 solution.
        boundary_point : `numpy.ndarray`
            The point that lies on the decision boundary
            (or an overshooted version).
        normal : `numpy.ndarray`
            The normal of the decision boundary at the boundary point.
        min_ : `numpy.ndarray`
            The minimum allowed input values.
        max_ : int
            The maximum allowed input values.

        """

        coordinates = normal
        normal_vec = normal.flatten()
        boundary_point_vec = boundary_point.flatten()
        initial_point_vec = initial_point.flatten()

        # Fit the initial point to the affine hyperplane and get the sign
        f_k = np.dot(normal_vec, initial_point_vec - boundary_point_vec)
        sign_true = np.sign(f_k)
        current_sign = sign_true

        perturbed = initial_point
        while current_sign == sign_true and np.count_nonzero(coordinates) > 0:

            # Fit the current point to the hyperplane.
            f_k = np.dot(normal_vec, perturbed.flatten() - boundary_point_vec)
            f_k = f_k + (1e-3 * sign_true)  # Avoid numerical instabilities

            # Compute the L1 projection (perturbation) of the current point
            # towards the direction of the maximum
            # absolute value
            mask = np.zeros_like(coordinates)
            mask[np.unravel_index(np.argmax(np.absolute(coordinates)),
                                  coordinates.shape)] = 1

            perturbation = max(abs(f_k) / np.amax(np.absolute(coordinates)),
                               1e-4) * mask * np.sign(coordinates)

            # Apply the perturbation
            perturbed = perturbed + perturbation
            perturbed = np.clip(perturbed, min_, max_)

            # Fit the point to the (unbiased) hyperplane and get the sign
            f_k = np.dot(normal_vec, perturbed.flatten() - boundary_point_vec)
            current_sign = np.sign(f_k)

            # Remove the used coordinate from the space of the available
            # coordinates
            coordinates[perturbation != 0] = 0

        # Return the l1 solution
        return perturbed - initial_point
