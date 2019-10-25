import numpy as np
from abc import abstractmethod
import logging
import warnings

from .base import Attack
from .base import generator_decorator
from .. import distances
from ..utils import crossentropy
from .. import nprng
from ..optimizers import AdamOptimizer
from ..optimizers import GDOptimizer


class IterativeProjectedGradientBaseAttack(Attack):
    """Base class for iterative (projected) gradient attacks.

    Concrete subclasses should implement as_generator, _gradient
    and _clip_perturbation.

    TODO: add support for other loss-functions, e.g. the CW loss function,
    see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """

    @abstractmethod
    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _create_optimizer(self, a, stepsize):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError

    def _get_mode_and_class(self, a):
        # determine if the attack is targeted or not
        target_class = a.target_class
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(
        self,
        a,
        binary_search,
        epsilon,
        stepsize,
        iterations,
        random_start,
        return_early,
        gradient_args={},
    ):
        if not a.has_gradient():
            warnings.warn(
                "applied gradient-based attack to model that"
                " does not provide gradients"
            )
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)

        if binary_search:
            if isinstance(binary_search, bool):
                k = 20
            else:
                k = int(binary_search)
            yield from self._run_binary_search(
                a,
                epsilon,
                stepsize,
                iterations,
                random_start,
                targeted,
                class_,
                return_early,
                k=k,
                gradient_args=gradient_args,
            )
            return
        else:
            optimizer = self._create_optimizer(a, stepsize)

            success = yield from self._run_one(
                a,
                epsilon,
                optimizer,
                iterations,
                random_start,
                targeted,
                class_,
                return_early,
                gradient_args,
            )
            return success

    def _run_binary_search(
        self,
        a,
        epsilon,
        stepsize,
        iterations,
        random_start,
        targeted,
        class_,
        return_early,
        k,
        gradient_args,
    ):

        factor = stepsize / epsilon

        def try_epsilon(epsilon):
            stepsize = factor * epsilon
            optimizer = self._create_optimizer(a, stepsize)

            success = yield from self._run_one(
                a,
                epsilon,
                optimizer,
                iterations,
                random_start,
                targeted,
                class_,
                return_early,
                gradient_args,
            )
            return success

        for i in range(k):
            success = yield from try_epsilon(epsilon)
            if success:
                logging.info("successful for eps = {}".format(epsilon))
                break
            logging.info("not successful for eps = {}".format(epsilon))
            epsilon = epsilon * 1.5
        else:
            logging.warning("exponential search failed")
            return

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            success = yield from try_epsilon(epsilon)
            if success:
                good = epsilon
                logging.info("successful for eps = {}".format(epsilon))
            else:
                bad = epsilon
                logging.info("not successful for eps = {}".format(epsilon))

    def _run_one(
        self,
        a,
        epsilon,
        optimizer,
        iterations,
        random_start,
        targeted,
        class_,
        return_early,
        gradient_args,
    ):
        min_, max_ = a.bounds()
        s = max_ - min_

        original = a.unperturbed.copy()

        if random_start:
            # using uniform noise even if the perturbation clipping uses
            # a different norm because cleverhans does it the same way
            noise = nprng.uniform(-epsilon * s, epsilon * s, original.shape).astype(
                original.dtype
            )
            x = original + self._clip_perturbation(a, noise, epsilon)
            strict = False  # because we don't enforce the bounds here
        else:
            x = original
            strict = True

        success = False
        for _ in range(iterations):
            gradient = yield from self._gradient(
                a, x, class_, strict=strict, gradient_args=gradient_args
            )
            # non-strict only for the first call and
            # only if random_start is True
            strict = True
            if not targeted:
                gradient = -gradient

            # untargeted: gradient ascent on cross-entropy to original class
            # targeted: gradient descent on cross-entropy to target class
            x = x + optimizer(gradient)

            x = original + self._clip_perturbation(a, x - original, epsilon)

            x = np.clip(x, min_, max_)

            logits, is_adversarial = yield from a.forward_one(x)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                if targeted:
                    ce = crossentropy(a.original_class, logits)
                    logging.debug(
                        "crossentropy to {} is {}".format(a.original_class, ce)
                    )
                ce = crossentropy(class_, logits)
                logging.debug("crossentropy to {} is {}".format(class_, ce))
            if is_adversarial:
                if return_early:
                    return True
                else:
                    success = True
        return success


class GDOptimizerMixin(object):
    def _create_optimizer(self, a, stepsize):
        return GDOptimizer(stepsize)


class AdamOptimizerMixin(object):
    def _create_optimizer(self, a, stepsize):
        return AdamOptimizer(a.unperturbed.shape, stepsize)


class LinfinityGradientMixin(object):
    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        gradient = yield from a.gradient_one(x, class_, strict=strict)
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class SparseL1GradientMixin(object):
    """Calculates a sparse L1 gradient introduced in [1]_.

         References
        ----------
        .. [1] Florian Tramèr, Dan Boneh,
               "Adversarial Training and Robustness for Multiple Perturbations",
               https://arxiv.org/abs/1904.13000

        """

    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        q = gradient_args["q"]

        gradient = yield from a.gradient_one(x, class_, strict=strict)

        # make gradient sparse
        abs_grad = np.abs(gradient)
        gradient_percentile_mask = abs_grad <= np.percentile(abs_grad.flatten(), q)
        e = np.sign(gradient)
        e[gradient_percentile_mask] = 0

        # using mean to make range of epsilons comparable to Linf
        normalization = np.mean(np.abs(e))

        gradient = e / normalization
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L1GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        gradient = yield from a.gradient_one(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient_norm = np.mean(np.abs(gradient))
        gradient_norm = max(1e-12, gradient_norm)
        gradient = gradient / gradient_norm
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L2GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        gradient = yield from a.gradient_one(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient_norm = np.sqrt(np.mean(np.square(gradient)))
        gradient_norm = max(1e-12, gradient_norm)
        gradient = gradient / gradient_norm
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class LinfinityClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        min_, max_ = a.bounds()
        s = max_ - min_
        clipped = np.clip(perturbation, -epsilon * s, epsilon * s)
        return clipped


class L1ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.mean(np.abs(perturbation))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class L2ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.sqrt(np.mean(np.square(perturbation)))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class LinfinityDistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.Linfinity):
            logging.warning(
                "Running an attack that tries to minimize the"
                " Linfinity norm of the perturbation without"
                " specifying foolbox.distances.Linfinity as"
                " the distance metric might lead to suboptimal"
                " results."
            )


class L1DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MAE):
            logging.warning(
                "Running an attack that tries to minimize the"
                " L1 norm of the perturbation without"
                " specifying foolbox.distances.MAE as"
                " the distance metric might lead to suboptimal"
                " results."
            )


class L2DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MSE):
            logging.warning(
                "Running an attack that tries to minimize the"
                " L2 norm of the perturbation without"
                " specifying foolbox.distances.MSE as"
                " the distance metric might lead to suboptimal"
                " results."
            )


class LinfinityBasicIterativeAttack(
    LinfinityGradientMixin,
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Basic Iterative Method introduced in [1]_.

    This attack is also known as Projected Gradient
    Descent (PGD) (without random start) or FGMS^k.

    References
    ----------
    .. [1] Alexey Kurakin, Ian Goodfellow, Samy Bengio,
           "Adversarial examples in the physical world",
            https://arxiv.org/abs/1607.02533

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


BasicIterativeMethod = LinfinityBasicIterativeAttack
BIM = BasicIterativeMethod


class L1BasicIterativeAttack(
    L1GradientMixin,
    L1ClippingMixin,
    L1DistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """Modified version of the Basic Iterative Method
    that minimizes the L1 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


class SparseL1BasicIterativeAttack(
    SparseL1GradientMixin,
    L1ClippingMixin,
    L1DistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """Sparse version of the Basic Iterative Method
    that minimizes the L1 distance introduced in [1]_.

     References
    ----------
    .. [1] Florian Tramèr, Dan Boneh,
           "Adversarial Training and Robustness for Multiple Perturbations",
           https://arxiv.org/abs/1904.13000

    .. seealso:: :class:`L1BasicIterativeAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        q=80.0,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Sparse version of a gradient-based attack that minimizes the
        L1 distance.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        q : float
            Relative percentile to make gradients sparse (must be in [0, 100))
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        assert 0 <= q < 100.0, "`q` must be in [0, 100)."

        yield from self._run(
            a,
            binary_search,
            epsilon,
            stepsize,
            iterations,
            random_start,
            return_early,
            gradient_args={"q": q},
        )


class L2BasicIterativeAttack(
    L2GradientMixin,
    L2ClippingMixin,
    L2DistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """Modified version of the Basic Iterative Method
    that minimizes the L2 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


class ProjectedGradientDescentAttack(
    LinfinityGradientMixin,
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Projected Gradient Descent Attack
    introduced in [1]_ without random start.

    When used without a random start, this attack
    is also known as Basic Iterative Method (BIM)
    or FGSM^k.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. seealso::

       :class:`LinfinityBasicIterativeAttack` and
       :class:`RandomStartProjectedGradientDescentAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.01,
        iterations=40,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


ProjectedGradientDescent = ProjectedGradientDescentAttack
PGD = ProjectedGradientDescent


class RandomStartProjectedGradientDescentAttack(
    LinfinityGradientMixin,
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Projected Gradient Descent Attack
    introduced in [1]_ with random start.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.01,
        iterations=40,
        random_start=True,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


RandomProjectedGradientDescent = RandomStartProjectedGradientDescentAttack
RandomPGD = RandomProjectedGradientDescent


class MomentumIterativeAttack(
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Momentum Iterative Method attack
    introduced in [1]_. It's like the Basic
    Iterative Method or Projected Gradient
    Descent except that it uses momentum.

    References
    ----------
    .. [1] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su,
           Jun Zhu, Xiaolin Hu, Jianguo Li, "Boosting Adversarial
           Attacks with Momentum",
           https://arxiv.org/abs/1710.06081

    """

    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        # get current gradient
        gradient = yield from a.gradient_one(x, class_, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = self._decay_factor * self._momentum_history + gradient

        # use history
        gradient = self._momentum_history
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient

    def _run_one(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._momentum_history = 0
        success = yield from super(MomentumIterativeAttack, self)._run_one(
            *args, **kwargs
        )
        return success

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.06,
        iterations=10,
        decay_factor=1.0,
        random_start=False,
        return_early=True,
    ):

        """Momentum-based iterative gradient attack known as
        Momentum Iterative Method.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        decay_factor : float
            Decay factor used by the momentum term.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        self._decay_factor = decay_factor

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


MomentumIterativeMethod = MomentumIterativeAttack


class AdamL1BasicIterativeAttack(
    L1GradientMixin,
    L1ClippingMixin,
    L1DistanceCheckMixin,
    AdamOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """Modified version of the Basic Iterative Method
    that minimizes the L1 distance using the Adam optimizer.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

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
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


class AdamL2BasicIterativeAttack(
    L2GradientMixin,
    L2ClippingMixin,
    L2DistanceCheckMixin,
    AdamOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """Modified version of the Basic Iterative Method
    that minimizes the L2 distance using the Adam optimizer.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.05,
        iterations=10,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

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
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


class AdamProjectedGradientDescentAttack(
    LinfinityGradientMixin,
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    AdamOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Projected Gradient Descent Attack
    introduced in [1]_, [2]_ without random start using the Adam optimizer.

    When used without a random start, this attack
    is also known as Basic Iterative Method (BIM)
    or FGSM^k.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. [2] Nicholas Carlini, David Wagner: "Towards Evaluating the
           Robustness of Neural Networks", https://arxiv.org/abs/1608.04644

    .. seealso::

       :class:`LinfinityBasicIterativeAttack` and
       :class:`RandomStartProjectedGradientDescentAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.01,
        iterations=40,
        random_start=False,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

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
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


AdamProjectedGradientDescent = AdamProjectedGradientDescentAttack
AdamPGD = AdamProjectedGradientDescent


class AdamRandomStartProjectedGradientDescentAttack(
    LinfinityGradientMixin,
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    AdamOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):

    """The Projected Gradient Descent Attack
    introduced in [1]_, [2]_ with random start using the Adam optimizer.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. [2] Nicholas Carlini, David Wagner: "Towards Evaluating the
           Robustness of Neural Networks", https://arxiv.org/abs/1608.04644

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        binary_search=True,
        epsilon=0.3,
        stepsize=0.01,
        iterations=40,
        random_start=True,
        return_early=True,
    ):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

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
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )


AdamRandomProjectedGradientDescent = (
    AdamRandomStartProjectedGradientDescentAttack
)  # noqa: E501

AdamRandomPGD = AdamRandomProjectedGradientDescent
